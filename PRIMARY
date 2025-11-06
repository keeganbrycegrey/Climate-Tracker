"""
Streamlit Climate Data Analyzer (Philippines-focused)

Single-file Streamlit app ready for local testing and deployment.

Features:
- Drag & drop multiple XLS/XLSX uploads
- Automatic dataset-type detection (Anomaly, Timeseries, Heatplot, Correlation)
- Parsers for common sheet/column layouts
- Summary statistics + natural-language observations
- Interactive plots (Plotly)
- Optional cross-referencing with external datasets (NASA POWER)
- Exportable analysis (CSV / JSON)

Dependencies (pip):
streamlit, pandas, numpy, plotly, openpyxl, xlrd, requests, scipy

Run locally:
$ pip install -r requirements.txt
$ streamlit run streamlit_climate_app.py

Deploy: Streamlit Community Cloud, Heroku, or any container service.

"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import base64
import requests
from datetime import datetime
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Climate Data Analyzer — Philippines", layout="wide")

# ------------------------- Helpers -------------------------

def read_excel_all_sheets(uploaded_file):
    """Read an uploaded xls/xlsx into a dict of dataframes."""
    try:
        xls = pd.read_excel(uploaded_file, sheet_name=None)
        return xls
    except Exception as e:
        # try with engine fallback
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl")


def detect_dataset_type(name: str, dfs: dict) -> str:
    """Simple heuristics to detect dataset type using filename and sheet/col names."""
    n = name.lower()
    cols = []
    for k, df in dfs.items():
        cols += [c.lower() for c in df.columns.astype(str).tolist()]
    cols = list(set(cols))

    if "correlation" in n or "corr" in n or "correl" in " ".join(cols):
        return "correlation"
    if "heat" in n or "heatmap" in n or "heat_plot" in n or any("lat" in c and "lon" in c for c in cols):
        return "heat"
    if "timeseries" in n or "time" in " ".join(cols) or any(c in cols for c in ["year", "date", "time"]):
        return "timeseries"
    if "anomaly" in n or "anomal" in " ".join(cols) or any("anomaly" in c for c in cols):
        return "anomaly"
    # fallback
    return "unknown"


def summarize_anomaly(df: pd.DataFrame, col: str = None):
    """Return summary statistics and a short natural-language observation for anomaly-style tables."""
    # heuristics to find anomaly column
    if col is None:
        candidates = [c for c in df.columns if "anom" in c.lower() or "anomaly" in c.lower() or "delta" in c.lower()]
        if candidates:
            col = candidates[0]
        else:
            # fallback to numeric columns
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                return None, "No numeric anomaly-like column found."
            col = numeric[0]

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None, "Anomaly column detected but contains no numeric data."

    mean = series.mean()
    median = series.median()
    std = series.std()
    maxi = series.max()
    mini = series.min()

    # trend: if there's a 'year' or 'time' column, compute linear trend per year
    trend_text = ""
    year_cols = [c for c in df.columns if c.lower() in ("year", "time", "date")]
    if year_cols:
        year_series = pd.to_numeric(df[year_cols[0]], errors="coerce")
        mask = (~year_series.isna()) & (~series.isna())
        if mask.sum() > 2:
            slope, intercept, r_value, p_value, stderr = stats.linregress(year_series[mask], series[mask])
            trend_text = f"Estimated linear trend: {slope:.4f} °C per year (r={r_value:.2f}, p={p_value:.3f})."

    obs = (
        f"Mean anomaly: {mean:.3f} °C (median {median:.3f}; std {std:.3f}). "
        f"Range: {mini:.3f} to {maxi:.3f} °C. {trend_text}"
    )

    summary = {
        "mean": mean,
        "median": median,
        "std": std,
        "min": mini,
        "max": maxi,
        "n": int(series.count())
    }
    return summary, obs


def summarize_timeseries(df: pd.DataFrame, value_col: str = None, time_col: str = None):
    if value_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            return None, "No numeric column found for timeseries."
        value_col = numeric[0]
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in ("year", "time", "date")]
        time_col = candidates[0] if candidates else df.columns[0]

    s_time = pd.to_datetime(df[time_col], errors="coerce")
    s_val = pd.to_numeric(df[value_col], errors="coerce")
    mask = (~s_time.isna()) & (~s_val.isna())
    if mask.sum() < 3:
        return None, "Not enough paired time/value datapoints for timeseries analysis."

    # aggregate by year if many daily values
    if (s_time.dt.year.nunique() < 200) and (s_time.dt.time.nunique() > 1):
        df_agg = pd.DataFrame({"year": s_time.dt.year, "value": s_val}).groupby("year").mean().reset_index()
        x = df_agg["year"].astype(int)
        y = df_agg["value"]
    else:
        x = s_time.map(datetime.toordinal)
        y = s_val

    slope, intercept, r_value, p_value, stderr = stats.linregress(x.dropna(), y.dropna())
    obs = f"Linear fit slope {slope:.5g} (r={r_value:.3f}, p={p_value:.3f})."
    summary = {"slope": slope, "r": r_value, "p": p_value}
    return summary, obs


def summarize_heat(df: pd.DataFrame, lat_col: str = None, lon_col: str = None, value_col: str = None):
    # try to locate columns
    cols = [c.lower() for c in df.columns]
    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lon" in c.lower()]
    value_candidates = [c for c in df.columns if any(k in c.lower() for k in ["temp", "value", "anom", "anomaly"])]

    if not lat_candidates or not lon_candidates or not value_candidates:
        return None, "Could not detect lat/lon/value columns for heat plot."

    latc, lonc, valc = lat_candidates[0], lon_candidates[0], value_candidates[0]
    df_clean = df[[latc, lonc, valc]].dropna()
    if df_clean.empty:
        return None, "No geographic data after dropping NA."

    v_mean = df_clean[valc].mean()
    v_std = df_clean[valc].std()
    obs = f"Spatial field mean: {v_mean:.3f}, std: {v_std:.3f}. Data points: {len(df_clean)}."
    summary = {"mean": v_mean, "std": v_std, "n": int(len(df_clean))}
    return summary, obs


def summarize_correlation(df: pd.DataFrame, x_col: str = None, y_col: str = None):
    if x_col is None or y_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            return None, "Need at least two numeric columns for correlation."
        x_col, y_col = numeric[0], numeric[1]

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = (~x.isna()) & (~y.isna())
    if mask.sum() < 3:
        return None, "Not enough paired numeric points for correlation test."

    r, p = stats.pearsonr(x[mask], y[mask])
    obs = f"Pearson r = {r:.3f}, p-value = {p:.4f} (n={int(mask.sum())})."
    summary = {"r": r, "p": p, "n": int(mask.sum())}
    return summary, obs


# ---------------------- External data: NASA POWER ----------------------

NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/ann/json"

def fetch_nasa_power_point(lat, lon, start_year=1980, end_year=2020, parameters=["T2M"]):
    """Fetch annualized parameters for a point from NASA POWER. Returns dataframe or None."""
    params = ",".join(parameters)
    url = (
        f"{NASA_POWER_BASE}?start={start_year}&end={end_year}&latitude={lat}&longitude={lon}"
        f"&parameters={params}&community=ag&format=JSON"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "properties" not in j or "parameter" not in j["properties"]:
            return None
        pdata = j["properties"]["parameter"]
        # construct df by year
        years = sorted(int(y) for y in list(next(iter(pdata.values())).keys()))
        df = pd.DataFrame({"year": years})
        for p in pdata.keys():
            vals = [pdata[p][str(y)] for y in years]
            df[p] = vals
        return df
    except Exception as e:
        return None


# ------------------------- Streamlit UI -------------------------

st.title("Climate Data Analyzer — Philippines")
st.markdown(
    "Upload one or more XLS/XLSX files exported from climate analysis tools (CMIP6 outputs, custom analyses).\n"
    "The app will attempt to detect dataset types, compute summaries, produce plots, and optionally cross-reference external datasets (e.g., NASA POWER)."
)

uploaded_files = st.file_uploader("Drag and drop XLS/XLSX files here", type=["xls", "xlsx"], accept_multiple_files=True)

# Sidebar controls
st.sidebar.header("Cross-reference settings")
crossref = st.sidebar.checkbox("Enable cross-referencing with external datasets (NASA POWER)", value=False)
if crossref:
    st.sidebar.markdown("Enter a reference point (latitude, longitude) to fetch NASA POWER annual data for comparison.")
    ref_lat = st.sidebar.number_input("Latitude", value=12.8797, format="%.6f")
    ref_lon = st.sidebar.number_input("Longitude", value=121.7740, format="%.6f")
    nr_start = st.sidebar.number_input("NASA POWER start year", value=1980, step=1)
    nr_end = st.sidebar.number_input("NASA POWER end year", value=2020, step=1)


if not uploaded_files:
    st.info("No files uploaded yet — drag and drop your XLS/XLSX files into the upload box. See sidebar for cross-reference options.")
    st.stop()

# Container for results
results = []

for f in uploaded_files:
    with st.spinner(f"Processing {f.name}..."):
        try:
            sheets = read_excel_all_sheets(f)
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")
            continue

        ds_type = detect_dataset_type(f.name, sheets)

        st.subheader(f"File: {f.name} — detected type: {ds_type}")

        # show sheet names
        sheet_names = list(sheets.keys())
        st.write("Sheets:", sheet_names)

        # show first sheet preview
        first_sheet = sheets[sheet_names[0]]
        st.dataframe(first_sheet.head())

        # Analysis routing
        file_result = {"filename": f.name, "type": ds_type, "sheets": sheet_names, "analyses": {}}

        if ds_type == "anomaly":
            # choose best sheet with numeric data
            sheet_df = first_sheet
            summary, obs = summarize_anomaly(sheet_df)
            if summary is None:
                st.warning(obs)
            else:
                st.write("**Anomaly summary**")
                st.json(summary)
                st.markdown(f"**Observation:** {obs}")
                # small plot
                num_cols = sheet_df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    c = num_cols[0]
                    fig = px.histogram(sheet_df, x=c, nbins=30, marginal="box", title=f"Distribution of {c}")
                    st.plotly_chart(fig, use_container_width=True)
            file_result["analyses"]["anomaly"] = {"summary": summary, "obs": obs}

        elif ds_type == "timeseries":
            sheet_df = first_sheet
            summary, obs = summarize_timeseries(sheet_df)
            if summary is None:
                st.warning(obs)
            else:
                st.write("**Timeseries summary**")
                st.json(summary)
                st.markdown(f"**Observation:** {obs}")
                # timeseries plot
                tcol = [c for c in sheet_df.columns if c.lower() in ("year", "time", "date")]
                vcols = sheet_df.select_dtypes(include=[np.number]).columns.tolist()
                if tcol and vcols:
                    t = tcol[0]
                    v = vcols[0]
                    fig = px.line(sheet_df, x=t, y=v, title=f"Timeseries: {v} vs {t}")
                    st.plotly_chart(fig, use_container_width=True)
            file_result["analyses"]["timeseries"] = {"summary": summary, "obs": obs}

        elif ds_type == "heat":
            sheet_df = first_sheet
            summary, obs = summarize_heat(sheet_df)
            if summary is None:
                st.warning(obs)
            else:
                st.write("**Heatmap summary**")
                st.json(summary)
                st.markdown(f"**Observation:** {obs}")
                # scatter map
                latc = [c for c in sheet_df.columns if "lat" in c.lower()][0]
                lonc = [c for c in sheet_df.columns if "lon" in c.lower()][0]
                valc = [c for c in sheet_df.columns if any(k in c.lower() for k in ["temp", "value", "anom", "anomaly"])][0]
                fig = px.scatter_geo(sheet_df, lat=latc, lon=lonc, color=valc, hover_name=valc, title="Spatial field (scatter)")
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
            file_result["analyses"]["heat"] = {"summary": summary, "obs": obs}

        elif ds_type == "correlation":
            sheet_df = first_sheet
            summary, obs = summarize_correlation(sheet_df)
            if summary is None:
                st.warning(obs)
            else:
                st.write("**Correlation summary**")
                st.json(summary)
                st.markdown(f"**Observation:** {obs}")
                # scatter
                numcols = sheet_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numcols) >= 2:
                    fig = px.scatter(sheet_df, x=numcols[0], y=numcols[1], trendline="ols", title=f"{numcols[0]} vs {numcols[1]}")
                    st.plotly_chart(fig, use_container_width=True)
            file_result["analyses"]["correlation"] = {"summary": summary, "obs": obs}

        else:
            # try to do best-effort analyses on each sheet
            combined_obs = []
            for sn, sdf in sheets.items():
                # try anomaly
                summ, obs = summarize_anomaly(sdf)
                if summ:
                    combined_obs.append({"sheet": sn, "type": "anomaly", "summary": summ, "obs": obs})
                else:
                    summt, obst = summarize_timeseries(sdf)
                    if summt:
                        combined_obs.append({"sheet": sn, "type": "timeseries", "summary": summt, "obs": obst})
            if combined_obs:
                st.write("**Auto-detected analyses on sheets**")
                st.json(combined_obs)
                for c in combined_obs:
                    st.markdown(f"**{c['sheet']} ({c['type']})** — {c['obs']}")
                file_result["analyses"]["auto"] = combined_obs
            else:
                st.info("Could not auto-detect a known dataset type. Consider renaming file or ensure standard column names are present (Year, Anomaly, Lat, Lon).")

        # Cross-reference with NASA POWER if requested
        if crossref:
            with st.spinner("Fetching external reference data from NASA POWER..."):
                exdf = fetch_nasa_power_point(ref_lat, ref_lon, nr_start, nr_end, parameters=["T2M"])
                if exdf is None:
                    st.warning("NASA POWER fetch failed or returned no data.")
                else:
                    st.write("**External reference (NASA POWER) — annual T2M**")
                    st.dataframe(exdf.head())
                    # quick comparison if timeseries present
                    if ds_type in ("timeseries", "anomaly"):
                        # attempt to compare annual means
                        numcols = first_sheet.select_dtypes(include=[np.number]).columns.tolist()
                        tcols = [c for c in first_sheet.columns if c.lower() in ("year", "time", "date")]
                        if tcols and numcols:
                            tcol = tcols[0]
                            vcol = numcols[0]
                            # prepare left and right
                            left = first_sheet[[tcol, vcol]].dropna()
                            left = left.rename(columns={tcol: "year", vcol: "value"})
                            left["year"] = pd.to_datetime(left["year"], errors="coerce").dt.year.fillna(left["year"]).astype(int)
                            right = exdf.rename(columns={"T2M": "ref_T2M"})
                            merged = pd.merge(left, right, on="year", how="inner")
                            if not merged.empty:
                                st.write("**Comparison with NASA POWER (merged years)**")
                                st.dataframe(merged.head())
                                fig = px.line(merged, x="year", y=["value", "ref_T2M"], labels={"value": f"Uploaded ({vcol})", "ref_T2M": "NASA POWER T2M"})
                                st.plotly_chart(fig, use_container_width=True)
                                # simple correlation
                                r, p = stats.pearsonr(merged["value"], merged["ref_T2M"]) if len(merged) > 2 else (np.nan, np.nan)
                                st.markdown(f"Correlation with NASA POWER for overlapping years: r={r:.3f}, p={p:.4f}")
                                file_result["external_compare"] = {"merged_n": int(len(merged)), "r": float(r) if not np.isnan(r) else None, "p": float(p) if not np.isnan(p) else None}
                            else:
                                st.info("No overlapping years/indices found to compare with NASA POWER")
                        else:
                            st.info("Uploaded file lacks obvious time/value columns — cannot meaningfully compare to NASA POWER.")

        # allow export of analysis for this file
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(f"Export analysis for {f.name}"):
                payload = json.dumps(file_result, default=str, indent=2)
                b64 = base64.b64encode(payload.encode()).decode()
                href = f"data:application/json;base64,{b64}"
                st.markdown(f"[Download JSON]({href})")

        results.append(file_result)

# Final combined export
if results:
    if st.button("Export combined analysis (all files)"):
        payload = json.dumps(results, default=str, indent=2)
        b64 = base64.b64encode(payload.encode()).decode()
        href = f"data:application/json;base64,{b64}"
        st.markdown(f"[Download combined JSON]({href})")

st.caption("App generated by assistant — customize the detection heuristics and interpretation rules for your specific datasets. For production, add authentication, logging, and larger-file streaming support.")
