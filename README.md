# Strategic Fusion Dashboard (Problem Statement 1)

This project is a functional web dashboard for fusing multi-source intelligence:
- OSINT from mock JSON (and optional MongoDB + AWS S3 connectors)
- HUMINT via manual CSV/JSON upload
- IMINT via drag-and-drop JPG/JPEG upload

All nodes are visualized on a single map with interactive hover popups showing metadata and image preview.

## Features delivered

1. **Automated OSINT retrieval**
   - Mock OSINT auto-load from `data/osint_mock.json`
   - Optional live pulls from MongoDB and AWS S3 (if credentials/endpoints are available)

2. **Manual ingestion**
   - Upload HUMINT files (`.csv` / `.json`)
   - Drag-and-drop IMINT files (`.jpg` / `.jpeg`) with configurable coordinates

3. **Unified terrain map**
   - Satellite terrain basemap
   - Geospatial markers based on latitude/longitude
   - Color-coding by intelligence type (OSINT, HUMINT, IMINT)

4. **Hover-and-view workflow**
   - Hover on markers to inspect metadata
   - IMINT entries show image preview directly in tooltip

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow

1. Launch app.
2. Keep **Load mock OSINT JSON** enabled.
3. Upload `data/humint_sample.csv`.
4. Upload one or more JPG/JPEG images in IMINT section.
5. Hover markers on the map to validate metadata and image popup behavior.

## Optional cloud/database connectors

- **MongoDB:** provide URI, database, and collection; click **Fetch MongoDB OSINT**.
- **S3:** provide bucket + prefix containing JSON arrays; click **Fetch S3 OSINT**.

If these systems are unavailable during demo, the app still remains fully functional with mock + manual ingestion.
