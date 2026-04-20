import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="LoRa Telemetry Reconstruction Dashboard", layout="wide")

EXPECTED_FIELDS = [
    "node_id",
    "timestamp",
    "latitude",
    "longitude",
]

OUTPUT_FIELDS = [
    "node_id",
    "timestamp",
    "latitude",
    "longitude",
    "status",
    "confidence",
    "source_packet",
    "ingest_note",
]


def parse_packet(raw: str) -> Tuple[Optional[dict], str]:
    cleaned = (raw or "").strip()
    if not cleaned:
        return None, "Empty packet"
    try:
        parsed = json.loads(cleaned)
        return parsed, "Valid JSON"
    except json.JSONDecodeError:
        pass

    # Attempt light auto-correction for common LoRa truncation errors.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end == -1:
        cleaned = cleaned[start:] + "}"
    elif start == -1 and end != -1:
        cleaned = "{" + cleaned[: end + 1]
    elif start != -1 and end != -1 and start < end:
        cleaned = cleaned[start : end + 1]
    elif ":" in cleaned and "{" not in cleaned:
        cleaned = "{" + cleaned + "}"

    cleaned = cleaned.replace("'", '"')
    try:
        parsed = json.loads(cleaned)
        return parsed, "Auto-corrected malformed packet"
    except json.JSONDecodeError:
        return None, "Broken packet (unrecoverable)"


def normalize_packet(parsed: dict) -> Optional[dict]:
    row = {}
    node = parsed.get("node_id") or parsed.get("id") or parsed.get("node") or parsed.get("device")
    lat = parsed.get("latitude") if "latitude" in parsed else parsed.get("lat")
    lon = parsed.get("longitude") if "longitude" in parsed else parsed.get("lon", parsed.get("lng"))
    ts = parsed.get("timestamp") or parsed.get("time") or datetime.utcnow().isoformat()
    row["node_id"] = str(node) if node is not None else None
    row["timestamp"] = ts
    row["latitude"] = pd.to_numeric(lat, errors="coerce")
    row["longitude"] = pd.to_numeric(lon, errors="coerce")
    if row["node_id"] is None:
        return None
    return row


def estimate_next(history: List[dict], requested_ts: pd.Timestamp) -> Optional[dict]:
    valid = [h for h in history if pd.notna(h["latitude"]) and pd.notna(h["longitude"])]
    if len(valid) < 1:
        return None
    if len(valid) == 1:
        anchor = valid[-1]
        return {
            "latitude": float(anchor["latitude"]),
            "longitude": float(anchor["longitude"]),
            "confidence": 0.55,
        }

    p1 = valid[-2]
    p2 = valid[-1]
    t1 = p1["ts"]
    t2 = p2["ts"]
    if pd.isna(t1) or pd.isna(t2):
        return None
    dt_hist = max((t2 - t1).total_seconds(), 1.0)
    dt_now = max((requested_ts - t2).total_seconds(), 1.0)
    ratio = dt_now / dt_hist

    vel_lat = float(p2["latitude"] - p1["latitude"])
    vel_lon = float(p2["longitude"] - p1["longitude"])
    predicted_lat = float(p2["latitude"] + (vel_lat * ratio))
    predicted_lon = float(p2["longitude"] + (vel_lon * ratio))

    # Alpha-beta style smoothing to avoid sharp jumps.
    alpha = 0.75
    smooth_lat = alpha * predicted_lat + (1 - alpha) * float(p2["latitude"])
    smooth_lon = alpha * predicted_lon + (1 - alpha) * float(p2["longitude"])
    decay = min(0.35, 0.07 * ratio)
    confidence = max(0.45, 0.80 - decay)
    return {
        "latitude": smooth_lat,
        "longitude": smooth_lon,
        "confidence": round(confidence, 3),
    }


def reconstruct_stream(raw_packets: List[str]) -> pd.DataFrame:
    if not raw_packets:
        return pd.DataFrame(columns=OUTPUT_FIELDS + ["ts"])
    node_history: Dict[str, List[dict]] = {}
    rows: List[dict] = []

    for raw in raw_packets:
        parsed, note = parse_packet(raw)
        if parsed is None:
            rows.append(
                {
                    "node_id": "unknown",
                    "timestamp": datetime.utcnow().isoformat(),
                    "latitude": None,
                    "longitude": None,
                    "status": "dropped",
                    "confidence": 0.0,
                    "source_packet": raw,
                    "ingest_note": note,
                    "ts": pd.NaT,
                }
            )
            continue

        normalized = normalize_packet(parsed)
        if normalized is None:
            rows.append(
                {
                    "node_id": "unknown",
                    "timestamp": parsed.get("timestamp", datetime.utcnow().isoformat()),
                    "latitude": None,
                    "longitude": None,
                    "status": "dropped",
                    "confidence": 0.0,
                    "source_packet": raw,
                    "ingest_note": "Missing node_id",
                    "ts": pd.NaT,
                }
            )
            continue

        node_id = normalized["node_id"]
        ts = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
        if pd.isna(ts):
            ts = pd.Timestamp.utcnow()
        normalized["ts"] = ts
        history = node_history.setdefault(node_id, [])

        if pd.notna(normalized["latitude"]) and pd.notna(normalized["longitude"]):
            rows.append(
                {
                    "node_id": node_id,
                    "timestamp": ts.isoformat(),
                    "latitude": float(normalized["latitude"]),
                    "longitude": float(normalized["longitude"]),
                    "status": "verified",
                    "confidence": 0.95 if note == "Valid JSON" else 0.85,
                    "source_packet": raw,
                    "ingest_note": note,
                    "ts": ts,
                }
            )
            history.append(normalized)
            continue

        estimate = estimate_next(history, ts)
        if estimate is None:
            rows.append(
                {
                    "node_id": node_id,
                    "timestamp": ts.isoformat(),
                    "latitude": None,
                    "longitude": None,
                    "status": "dropped",
                    "confidence": 0.0,
                    "source_packet": raw,
                    "ingest_note": "Broken + insufficient history",
                    "ts": ts,
                }
            )
            continue

        rows.append(
            {
                "node_id": node_id,
                "timestamp": ts.isoformat(),
                "latitude": estimate["latitude"],
                "longitude": estimate["longitude"],
                "status": "estimated",
                "confidence": estimate["confidence"],
                "source_packet": raw,
                "ingest_note": "Predicted via historical buffer",
                "ts": ts,
            }
        )
        history.append(
            {
                "node_id": node_id,
                "timestamp": ts.isoformat(),
                "latitude": estimate["latitude"],
                "longitude": estimate["longitude"],
                "ts": ts,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_FIELDS + ["ts"])
    out = out.sort_values(["node_id", "ts"]).reset_index(drop=True)
    return out


def create_pdf_summary_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "LoRa Predictive Reconstruction - Summary")
    y -= 24
    pdf.setFont("Helvetica", 10)
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.drawString(40, y, f"Generated: {generated}")
    y -= 24
    pdf.drawString(40, y, f"Total reconstructed packets: {len(df)}")
    y -= 16
    verified = int((df["status"] == "verified").sum()) if not df.empty else 0
    estimated = int((df["status"] == "estimated").sum()) if not df.empty else 0
    dropped = int((df["status"] == "dropped").sum()) if not df.empty else 0
    pdf.drawString(40, y, f"Verified: {verified} | Estimated: {estimated} | Dropped: {dropped}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y, "Recent Telemetry Records")
    y -= 18
    pdf.setFont("Helvetica", 9)
    recent_rows = df.sort_values("ts", ascending=False).head(12)
    for _, row in recent_rows.iterrows():
        line = (
            f"{row['timestamp'][:19]} | {row['node_id']} | {row['status']} | conf={row['confidence']}"
        )
        pdf.drawString(40, y, line)
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 9)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


if "reconstructed" not in st.session_state:
    st.session_state.reconstructed = pd.DataFrame(columns=OUTPUT_FIELDS + ["ts"])

st.title("Predictive Reconstruction of Fragmented LoRa Telemetry")
st.caption("Problem Statement 2: Broken packet handling + historical prediction + visual continuity")

with st.sidebar:
    st.header("LoRa Ingestion")
    st.markdown("Input can be valid JSON packets, partial packets, or malformed JSON lines.")
    upload_file = st.file_uploader("Upload packet file (TXT/CSV/JSON)", type=["txt", "csv", "json"])
    run_upload = st.button("Process Uploaded Stream")
    use_demo = st.button("Load Demo Fragmented Stream")
    clear_data = st.button("Clear Session Data")

col1, col2 = st.columns([2, 1])

if clear_data:
    st.session_state.reconstructed = pd.DataFrame(columns=OUTPUT_FIELDS + ["ts"])
    st.success("Session cleared.")

if use_demo:
    demo_packets = [
        '{"node_id":"A1","timestamp":"2026-04-20T09:00:00Z","latitude":28.6128,"longitude":77.2295}',
        '{"node_id":"A1","timestamp":"2026-04-20T09:01:00Z","latitude":28.6132,"longitude":77.2302}',
        '{"node_id":"A1","timestamp":"2026-04-20T09:02:00Z","lat":28.6136,"lon":77.2310',
        '{"node_id":"B2","timestamp":"2026-04-20T09:00:00Z","latitude":28.5355,"longitude":77.3910}',
        '{"node_id":"B2","timestamp":"2026-04-20T09:01:00Z","latitude":28.5361,"longitude":77.3921}',
        '{"node_id":"B2","timestamp":"2026-04-20T09:02:00Z","latitude":,"longitude":77.3930}',
        '{"node_id":"B2","timestamp":"2026-04-20T09:03:00Z","latitude":28.5373,"longitude":77.3942}',
        '{"node_id":"A1","timestamp":"2026-04-20T09:03:00Z","latitude":28.6141,"longitude":77.2319}',
    ]
    st.session_state.reconstructed = reconstruct_stream(demo_packets)
    st.success("Demo fragmented telemetry loaded.")

if run_upload and upload_file:
    try:
        blob = upload_file.read().decode("utf-8", errors="ignore")
        if upload_file.name.lower().endswith(".json"):
            parsed_json = json.loads(blob)
            if isinstance(parsed_json, list):
                packet_lines = [json.dumps(item) for item in parsed_json]
            else:
                packet_lines = [json.dumps(parsed_json)]
        else:
            packet_lines = [line.strip() for line in blob.splitlines() if line.strip()]
        st.session_state.reconstructed = reconstruct_stream(packet_lines)
        st.success(f"Processed packets: {len(packet_lines)}")
    except Exception as exc:
        st.error(f"Failed to parse uploaded file: {exc}")

data = st.session_state.reconstructed.copy()
if data.empty:
    st.info("No telemetry yet. Load demo stream or upload a packet file.")
    st.stop()

map_data = data.dropna(subset=["latitude", "longitude"]).copy()
status_color = {"verified": [0, 210, 120], "estimated": [255, 195, 0], "dropped": [255, 80, 80]}
map_data["color"] = map_data["status"].apply(lambda s: status_color.get(s, [200, 200, 200]))
map_data["dot_radius"] = map_data["confidence"].fillna(0).apply(lambda c: 60 + (c * 120))
map_data["tooltip_html"] = map_data.apply(
    lambda row: (
        f"<b>Node:</b> {row['node_id']}<br/>"
        f"<b>Status:</b> {row['status']}<br/>"
        f"<b>Time:</b> {row['timestamp']}<br/>"
        f"<b>Confidence:</b> {row['confidence']}<br/>"
        f"<b>Note:</b> {row['ingest_note']}"
    ),
    axis=1,
)

with col1:
    st.subheader("Movement Continuity Map (Verified vs Estimated)")
    if map_data.empty:
        st.warning("No valid geospatial points to render.")
        st.stop()

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_radius="dot_radius",
        get_fill_color="color",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_data["latitude"].mean()),
        longitude=float(map_data["longitude"].mean()),
        zoom=10,
        pitch=35,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"html": "{tooltip_html}", "style": {"backgroundColor": "black", "color": "white"}},
        )
    )

with col2:
    st.subheader("Analytics")
    st.metric("Total Nodes", int(len(data)))
    st.metric("Verified Points", int((data["status"] == "verified").sum()))
    st.metric("Estimated Points", int((data["status"] == "estimated").sum()))
    st.metric("Dropped Packets", int((data["status"] == "dropped").sum()))

    st.subheader("Telemetry Table")
    display_cols = [
        "node_id",
        "status",
        "latitude",
        "longitude",
        "timestamp",
        "confidence",
        "ingest_note",
    ]
    st.dataframe(data[display_cols], use_container_width=True)

st.subheader("Downloadable Reconstruction Report")
report_cols = [
    "node_id",
    "status",
    "latitude",
    "longitude",
    "timestamp",
    "confidence",
    "ingest_note",
    "source_packet",
]
csv_bytes = data[report_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV Report",
    data=csv_bytes,
    file_name="lora_reconstruction_report.csv",
    mime="text/csv",
)

pdf_bytes = create_pdf_summary_bytes(data)
st.download_button(
    label="Download PDF Summary",
    data=pdf_bytes,
    file_name="lora_reconstruction_summary.pdf",
    mime="application/pdf",
)
