import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# ============================================================
# FILE LOADING
# ============================================================

def load_excel_file(uploaded_file):
    try:
        return pd.read_excel(uploaded_file)
    except Exception:
        try:
            return pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception:
            return None

# ============================================================
# DATASET TYPE DETECTION
# ============================================================

def detect_dataset_type(df):
    cols = [c.lower() for c in df.columns]

    if any("anomaly" in c for c in cols) and any("time" in c for c in cols):
        return "Projected Timeseries Anomaly"
    if any("anomaly" in c for c in cols) and any("mean" in c for c in cols):
        return "Projected Anomaly"
    if any("heat" in c for c in cols) or (len(df.columns) > 10 and df.shape[0] > 10):
        return "Projected Heat Plot"
    if any("correlation" in c for c in cols) or "corr" in cols:
        return "Correlation Dataset"

    return "Unknown Dataset"

# ============================================================
# NASA POWER API
# ============================================================

def nasa_power_query(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M&community=AG&"
        f"longitude={lon}&latitude={lat}&start=20200101&end=20201231&format=JSON"
    )
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        t2m = list(data["properties"]["parameter"]["T2M"].values())
        return np.array(t2m)
    except Exception:
        return None

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Climate Data Processor — Philippines")
st.write("Drag and drop XLS/XLSX climate datasets to generate observations.")

uploaded_files = st.file_uploader(
    "Upload datasets",
    type=["xls", "xlsx"],
    accept_multiple_files=True
)

enable_nasa = st.checkbox("Cross-reference with NASA POWER (optional)")
if enable_nasa:
    lat = st.number_input("Latitude", value=14.6)
    lon = st.number_input("Longitude", value=121.0)

results = []

# ============================================================
# PROCESS FILES
# ============================================================

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"File: {file.name}")

        df = load_excel_file(file)
        if df is None:
            st.error("Unable to read file.")
            continue

        dtype = detect_dataset_type(df)
        st.write(f"**Detected Dataset Type:** {dtype}")
        st.dataframe(df.head())

        # ---------------------------
        # CORRELATION MATRIX
        # ---------------------------
        if "Correlation" in dtype:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()

            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

            results.append({
                "file": file.name,
                "type": dtype,
                "correlation": corr_matrix.to_dict()
            })

        # ---------------------------
        # TIMESERIES ANOMALY
        # ---------------------------
        elif "Timeseries" in dtype:
            time_col = df.columns[0]
            val_col = df.columns[1]

            fig = px.line(df, x=time_col, y=val_col, title="Timeseries Anomaly")
            st.plotly_chart(fig, use_container_width=True)

            results.append({"file": file.name, "type": dtype})

        # ---------------------------
        # HEAT PLOT
        # ---------------------------
        elif "Heat" in dtype:
            heat_df = df.select_dtypes(include=[np.number])

            fig = px.imshow(
                heat_df,
                title="Heat Plot",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

            results.append({"file": file.name, "type": dtype})

        # ---------------------------
        # MEAN ANOMALY
        # ---------------------------
        elif "Anomaly" in dtype:
            numeric = df.select_dtypes(include=[np.number])
            fig = px.bar(
                numeric.mean(),
                title="Average Anomaly"
            )
            st.plotly_chart(fig, use_container_width=True)

            results.append({"file": file.name, "type": dtype})

        # ======================================================
        # NASA POWER DATA
        # ======================================================
        if enable_nasa:
            nasa = nasa_power_query(lat, lon)
            if nasa is not None:
                st.write("NASA POWER data retrieved.")
                fig = px.line(nasa, title="NASA POWER T2M (Daily, 2020)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to retrieve NASA data.")

    st.success("Processing completed.")

    st.download_button(
        label="Download Summary (JSON)",
        data=str(results),
        file_name="results.json",
        mime="application/json"
    )
