```python
# STREAMLIT CLIMATE APP — DEBUGGED VERSION (NO SCIPY DEPENDENCY)
# Fully self-contained. Safe to copy-paste.
# Replace PRIMARY.py or use as app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

# ----------------------------------------------------
# UTILITIES
# ----------------------------------------------------

def load_excel_file(uploaded_file):
    try:
        return pd.read_excel(uploaded_file)
    except Exception:
        try:
            return pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception:
            return None


def detect_dataset_type(df):
    cols = [c.lower() for c in df.columns]

    if any("anomaly" in c for c in cols) and any("time" in c for c in cols):
        return "Projected Timeseries Anomaly"

    if any("anomaly" in c for c in cols) and any("mean" in c for c in cols):
        return "Projected Anomaly"

    if any("heat" in c for c in cols) or (len(df.columns) > 10 and df.shape[0] > 10):
        return "Projected Heat Plot"

    if any("correlation" in c for c in cols):
        return "Correlation Plot"

    return "Unknown Dataset"


def nasa_power_query(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M&community=AG&"
        f"longitude={lon}&latitude={lat}&start=20200101&end=20201231&format=JSON"
    )

    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        t2m = list(data["properties"]["parameter"]["T2M"].values())
        return np.array(t2m)
    except Exception:
        return None


# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------

st.title("Climate Data Processor — Philippines")
st.subheader("Drag-and-drop XLS/XLSX datasets for analysis")

uploaded = st.file_uploader(
    "Upload climate datasets (XLS/XLSX)", type=["xls", "xlsx"], accept_multiple_files=True
)

nasa_enable = st.checkbox("Cross-reference with NASA POWER (optional)")
if nasa_enable:
    lat = st.number_input("Latitude", value=14.6)
    lon = st.number_input("Longitude", value=121.0)

results = []

if uploaded:
    for file in uploaded:
        st.markdown(f"### File: **{file.name}**")

        df = load_excel_file(file)
        if df is None:
            st.error("Could not read this file.")
            continue

        dtype = detect_dataset_type(df)
        st.write(f"**Detected Type:** {dtype}")

        st.dataframe(df.head())

        # ------------------------------------------------------------
        # ANALYSIS LOGIC (Non-SciPy correlation)
        # ------------------------------------------------------------

        if "corr" in dtype.lower():
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr(method="pearson")

            fig = px.imshow(corr, title="Correlation Matrix", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            results.append({"file": file.name, "type": dtype, "corr": corr.to_dict()})

        elif "timeseries" in dtype.lower():
            time_col = df.columns[0]
            val_col = df.columns[1]

            fig = px.line(df, x=time_col, y=val_col, title="Timeseries Anomaly")
            st.plotly_chart(fig, use_container_width=True)
            results.append({"file": file.name, "type": dtype})

        elif "heat" in dtype.lower():
            fig = px.imshow(df.select_dtypes(include=[np.number]), title="Heat Plot", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            results.append({"file": file.name, "type": dtype})

        elif "anomaly" in dtype.lower():
            numeric_df = df.select_dtypes(include=[np.number])
            fig = px.bar(numeric_df.mean(), title="Mean Anomaly Values")
            st.plotly_chart(fig, use_container_width=True)
            results.append({"file": file.name, "type": dtype})

        # ------------------------------------------------------------
        # NASA POWER SECTION
        # ------------------------------------------------------------

        if nasa_enable:
            nasa = nasa_power_query(lat, lon)
            if nasa is not None:
                st.write("**NASA POWER Reference Data Loaded**")
                fig = px.line(y=nasa, title="NASA POWER T2M (Daily, 2020)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not load NASA POWER data.")

    st.success("Processing complete.")


# ----------------------------------------------------
# EXPORT BUTTON
# ----------------------------------------------------
if uploaded:
    export = st.download_button(
        label="Download Combined Results (JSON)",
        data=str(results),
        file_name="climate_results.json",
        mime="application/json",
    )
```}]}
