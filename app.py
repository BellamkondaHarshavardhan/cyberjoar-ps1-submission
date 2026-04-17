import base64
import io
import json
import math
from datetime import datetime
from typing import Dict, List

import pandas as pd
import pydeck as pdk
import streamlit as st

try:
    import boto3
except ImportError:
    boto3 = None

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None
    A4 = None

st.set_page_config(page_title="Strategic Fusion Dashboard", layout="wide")

REQUIRED = [
    "source",
    "intel_type",
    "title",
    "description",
    "latitude",
    "longitude",
    "timestamp",
    "image_data_uri",
]


def to_data_uri(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def normalize(df: pd.DataFrame, source: str, intel_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REQUIRED)
    df = df.rename(
        columns={
            "lat": "latitude",
            "lon": "longitude",
            "lng": "longitude",
            "name": "title",
            "details": "description",
            "time": "timestamp",
            "image": "image_data_uri",
            "image_url": "image_data_uri",
        }
    ).copy()
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = None
    df["source"] = df["source"].fillna(source)
    df["intel_type"] = df["intel_type"].fillna(intel_type)
    df["title"] = df["title"].fillna("Untitled Node")
    df["description"] = df["description"].fillna("No details supplied.")
    df["timestamp"] = df["timestamp"].fillna(datetime.utcnow().isoformat())
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["image_data_uri"] = df["image_data_uri"].fillna("")
    return df.dropna(subset=["latitude", "longitude"])[REQUIRED]


def load_mock_osint() -> pd.DataFrame:
    with open("data/osint_mock.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize(pd.DataFrame(data), "OSINT-MOCK", "OSINT")


def load_mongodb(uri: str, db: str, collection: str) -> pd.DataFrame:
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed. Install it from requirements.txt")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    docs: List[dict] = list(client[db][collection].find({}, {"_id": 0}))
    return normalize(pd.DataFrame(docs), "MongoDB", "OSINT")


def load_s3(bucket: str, prefix: str) -> pd.DataFrame:
    if boto3 is None:
        raise RuntimeError("boto3 is not installed. Install it from requirements.txt")
    s3 = boto3.client("s3")
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    rows = []
    for obj in objects:
        key = obj["Key"]
        if not key.lower().endswith(".json"):
            continue
        payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        rows.extend(json.loads(payload))
    return normalize(pd.DataFrame(rows), "S3", "OSINT")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    reliability = {
        "OSINT": 0.65,
        "HUMINT": 0.80,
        "IMINT": 0.90,
    }
    now = pd.Timestamp.utcnow()
    out = df.copy()
    out["ts"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["age_hours"] = ((now - out["ts"]).dt.total_seconds() / 3600).fillna(72).clip(lower=0, upper=168)
    out["source_reliability"] = out["intel_type"].map(reliability).fillna(0.55)
    out["recency_score"] = (1 - (out["age_hours"] / 168)).clip(lower=0)
    out["confidence_score"] = (0.7 * out["source_reliability"] + 0.3 * out["recency_score"]).round(3)
    out["corroboration_score"] = 0.3
    out["priority_score"] = (
        0.45 * out["source_reliability"] + 0.35 * out["recency_score"] + 0.20 * out["corroboration_score"]
    ).round(3)
    out["verification_status"] = out["confidence_score"].apply(
        lambda x: "verified" if x >= 0.8 else ("estimated" if x >= 0.6 else "unverified")
    )
    return out


def build_fused_nodes(df: pd.DataFrame, radius_m: float, window_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sorted_df = df.sort_values("ts").reset_index(drop=True).copy()
    used = set()
    clusters = []

    for idx in range(len(sorted_df)):
        if idx in used:
            continue
        base = sorted_df.loc[idx]
        members = [idx]
        used.add(idx)
        for j in range(idx + 1, len(sorted_df)):
            if j in used:
                continue
            cand = sorted_df.loc[j]
            if pd.isna(base["ts"]) or pd.isna(cand["ts"]):
                continue
            time_gap = abs((cand["ts"] - base["ts"]).total_seconds()) / 60
            if time_gap > window_minutes:
                continue
            dist = haversine_meters(base["latitude"], base["longitude"], cand["latitude"], cand["longitude"])
            if dist <= radius_m:
                members.append(j)
                used.add(j)

        c_df = sorted_df.loc[members]
        sources = sorted(c_df["intel_type"].unique().tolist())
        corroboration_bonus = min(1.0, 0.35 + (len(sources) - 1) * 0.25 + (len(c_df) - 1) * 0.1)
        priority = (
            0.45 * c_df["source_reliability"].mean()
            + 0.35 * c_df["recency_score"].mean()
            + 0.20 * corroboration_bonus
        )
        clusters.append(
            {
                "cluster_id": f"FUSION-{len(clusters)+1:03d}",
                "latitude": c_df["latitude"].mean(),
                "longitude": c_df["longitude"].mean(),
                "node_count": int(len(c_df)),
                "sources": ", ".join(sources),
                "avg_confidence": round(float(c_df["confidence_score"].mean()), 3),
                "priority_score": round(float(priority), 3),
                "fusion_reason": f"Co-located within {int(radius_m)}m and {window_minutes}min",
                "latest_timestamp": c_df["ts"].max().isoformat() if c_df["ts"].notna().any() else "",
            }
        )
    return pd.DataFrame(clusters)


def create_pdf_summary_bytes(nodes_df: pd.DataFrame, fused_df: pd.DataFrame) -> bytes:
    if canvas is None or A4 is None:
        raise RuntimeError("reportlab is not installed.")
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Strategic Fusion Dashboard - Intelligence Summary")
    y -= 24
    pdf.setFont("Helvetica", 10)
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.drawString(40, y, f"Generated: {generated}")
    y -= 24
    pdf.drawString(40, y, f"Total raw nodes: {len(nodes_df)}")
    y -= 16
    pdf.drawString(40, y, f"Total fused clusters: {len(fused_df)}")
    y -= 16
    if not nodes_df.empty:
        avg_conf = round(float(nodes_df["confidence_score"].mean()), 3)
        avg_priority = round(float(nodes_df["priority_score"].mean()), 3)
        pdf.drawString(40, y, f"Avg confidence: {avg_conf} | Avg priority: {avg_priority}")
        y -= 24

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y, "Top 10 High-Priority Nodes")
    y -= 18
    pdf.setFont("Helvetica", 9)
    top_nodes = nodes_df.sort_values("priority_score", ascending=False).head(10)
    for _, row in top_nodes.iterrows():
        line = (
            f"{row['title'][:35]} | {row['intel_type']} | priority={row['priority_score']} | "
            f"confidence={row['confidence_score']}"
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


if "all_data" not in st.session_state:
    st.session_state.all_data = pd.DataFrame(columns=REQUIRED)

st.title("Multi-Source Intelligence Fusion Dashboard")
st.caption("Problem Statement 1: OSINT + HUMINT + IMINT in one geospatial view")

with st.sidebar:
    st.header("Ingestion")
    load_mock = st.checkbox("Load mock OSINT JSON", value=True)
    st.subheader("MongoDB (Optional)")
    mongo_uri = st.text_input("Mongo URI", "mongodb://localhost:27017")
    mongo_db = st.text_input("Database", "fusion_db")
    mongo_col = st.text_input("Collection", "osint")
    mongo_btn = st.button("Fetch MongoDB OSINT")

    st.subheader("AWS S3 (Optional)")
    s3_bucket = st.text_input("S3 Bucket", "")
    s3_prefix = st.text_input("S3 Prefix", "osint/")
    s3_btn = st.button("Fetch S3 OSINT")

    st.subheader("HUMINT Upload")
    humint_file = st.file_uploader("Upload CSV/JSON", type=["csv", "json"])

    st.subheader("IMINT Upload (Drag & Drop)")
    imint_lat = st.number_input("IMINT Latitude", value=28.6139, format="%.6f")
    imint_lon = st.number_input("IMINT Longitude", value=77.2090, format="%.6f")
    imint_files = st.file_uploader(
        "Upload JPG/JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True
    )
    st.subheader("Advanced Controls")
    selected_types = st.multiselect("Filter Intelligence Type", ["OSINT", "HUMINT", "IMINT"], default=["OSINT", "HUMINT", "IMINT"])
    fusion_radius = st.slider("Fusion Radius (meters)", min_value=100, max_value=2000, value=500, step=50)
    fusion_window = st.slider("Fusion Time Window (minutes)", min_value=5, max_value=240, value=30, step=5)

col1, col2 = st.columns([2, 1])
new_frames = []

if load_mock:
    new_frames.append(load_mock_osint())

if mongo_btn:
    try:
        new_frames.append(load_mongodb(mongo_uri, mongo_db, mongo_col))
        st.success("MongoDB data loaded.")
    except Exception as exc:
        st.warning(f"MongoDB unavailable. Continuing without it. ({exc})")

if s3_btn:
    try:
        new_frames.append(load_s3(s3_bucket, s3_prefix))
        st.success("S3 data loaded.")
    except Exception as exc:
        st.warning(f"S3 unavailable. Continuing without it. ({exc})")

if humint_file:
    try:
        if humint_file.name.lower().endswith(".csv"):
            hum_df = pd.read_csv(humint_file)
        else:
            hum_df = pd.DataFrame(json.load(humint_file))
        new_frames.append(normalize(hum_df, "Manual Upload", "HUMINT"))
        st.success("HUMINT file ingested.")
    except Exception as exc:
        st.error(f"Failed to parse HUMINT file: {exc}")

if imint_files:
    rows = []
    for file in imint_files:
        uri = to_data_uri(file.read(), "image/jpeg")
        rows.append(
            {
                "source": "Manual Upload",
                "intel_type": "IMINT",
                "title": file.name,
                "description": "Uploaded satellite/field image",
                "latitude": imint_lat,
                "longitude": imint_lon,
                "timestamp": datetime.utcnow().isoformat(),
                "image_data_uri": uri,
            }
        )
    new_frames.append(normalize(pd.DataFrame(rows), "Manual Upload", "IMINT"))
    st.success(f"IMINT images ingested: {len(rows)}")

if new_frames:
    st.session_state.all_data = pd.concat([st.session_state.all_data] + new_frames, ignore_index=True)
    st.session_state.all_data = st.session_state.all_data.drop_duplicates(
        subset=["title", "latitude", "longitude", "timestamp"]
    )

data = st.session_state.all_data.copy()
if data.empty:
    st.info("No data points yet. Load mock data or upload files from the sidebar.")
    st.stop()

data = compute_scores(data)
if selected_types:
    data = data[data["intel_type"].isin(selected_types)].copy()

if data.empty:
    st.warning("No records match current filters.")
    st.stop()

valid_times = data["ts"].dropna()
if not valid_times.empty:
    min_time = valid_times.min().to_pydatetime()
    max_time = valid_times.max().to_pydatetime()
    selected_range = st.slider("Timeline Filter", min_value=min_time, max_value=max_time, value=(min_time, max_time))
    start_utc = pd.Timestamp(selected_range[0])
    end_utc = pd.Timestamp(selected_range[1])
    if start_utc.tzinfo is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")
    if end_utc.tzinfo is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")
    data = data[(data["ts"] >= start_utc) & (data["ts"] <= end_utc)].copy()

if data.empty:
    st.warning("No records in selected timeline range.")
    st.stop()

fused_df = build_fused_nodes(data, fusion_radius, fusion_window)
if not fused_df.empty:
    cluster_support = fused_df.set_index("cluster_id")["node_count"].to_dict()
    data = data.copy()
    data["corroboration_score"] = data["corroboration_score"].fillna(0.3)
    data["priority_score"] = (0.45 * data["source_reliability"] + 0.35 * data["recency_score"] + 0.20 * data["corroboration_score"]).round(3)

colors = {"OSINT": [0, 128, 255], "HUMINT": [0, 200, 120], "IMINT": [255, 90, 90]}
status_color = {"verified": [0, 220, 120], "estimated": [255, 195, 0], "unverified": [255, 80, 80]}
default_color = [240, 180, 30]
data["color"] = data["verification_status"].apply(lambda s: status_color.get(s, default_color))
data["dot_radius"] = (50 + (data["priority_score"] * 120)).astype(float)
data["tooltip_html"] = data.apply(
    lambda row: (
        f"<b>{row['intel_type']}</b> | {row['title']}<br/>"
        f"{row['description']}<br/>"
        f"Time: {row['timestamp']}<br/>"
        f"Confidence: {row['confidence_score']} | Priority: {row['priority_score']}<br/>"
        f"Status: {row['verification_status']}<br/>"
        + (f"<img src='{row['image_data_uri']}' width='220'/>" if row["image_data_uri"] else "")
    ),
    axis=1,
)

with col1:
    st.subheader("Unified Terrain Map")
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[longitude, latitude]",
        get_radius="dot_radius",
        get_fill_color="color",
        pickable=True,
    )
    layers = [layer]
    if not fused_df.empty:
        fused_df = fused_df.copy()
        fused_df["fusion_tooltip"] = fused_df.apply(
            lambda row: (
                f"<b>{row['cluster_id']}</b><br/>"
                f"Nodes: {row['node_count']} | Sources: {row['sources']}<br/>"
                f"Avg Confidence: {row['avg_confidence']} | Priority: {row['priority_score']}<br/>"
                f"{row['fusion_reason']}"
            ),
            axis=1,
        )
        fusion_layer = pdk.Layer(
            "ScatterplotLayer",
            data=fused_df,
            get_position="[longitude, latitude]",
            get_radius=180,
            get_fill_color=[230, 0, 255, 140],
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
        )
        layers.append(fusion_layer)

    view_state = pdk.ViewState(
        latitude=float(data["latitude"].mean()),
        longitude=float(data["longitude"].mean()),
        zoom=10,
        pitch=35,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "{tooltip_html}", "style": {"backgroundColor": "black", "color": "white"}},
        )
    )

with col2:
    st.subheader("Analytics")
    st.metric("Total Nodes", int(len(data)))
    st.metric("Fused Clusters", int(len(fused_df)))
    st.metric("Average Priority", round(float(data["priority_score"].mean()), 3))
    st.metric("Average Confidence", round(float(data["confidence_score"].mean()), 3))

    st.subheader("Node Table")
    display_cols = [
        "source",
        "intel_type",
        "title",
        "latitude",
        "longitude",
        "timestamp",
        "confidence_score",
        "priority_score",
        "verification_status",
    ]
    st.dataframe(data[display_cols], use_container_width=True)
    if not fused_df.empty:
        st.subheader("Fusion Clusters")
        st.dataframe(fused_df, use_container_width=True)

st.subheader("Downloadable Intelligence Report")
report_cols = [
    "source",
    "intel_type",
    "title",
    "description",
    "latitude",
    "longitude",
    "timestamp",
    "confidence_score",
    "priority_score",
    "verification_status",
]
csv_bytes = data[report_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV Summary",
    data=csv_bytes,
    file_name="intelligence_summary.csv",
    mime="text/csv",
)

if canvas is None:
    st.info("Install reportlab to enable PDF export: pip install reportlab")
else:
    pdf_bytes = create_pdf_summary_bytes(data, fused_df)
    st.download_button(
        label="Download PDF Summary",
        data=pdf_bytes,
        file_name="intelligence_summary.pdf",
        mime="application/pdf",
    )
