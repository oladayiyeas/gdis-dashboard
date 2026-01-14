"""
stats_panel.py
--------------------------------------------
Handles statistical summaries and visualization panels
for the open-source SCC Dashboard.

Compatible with datasets from Eti-Osa and Surulere LGAs
at both Grid Levels 1 and 2.

Author: [Your Name]
Version: 1.1
--------------------------------------------
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# ============================================================
# 1️⃣ Detect Numeric Columns
# ============================================================
def detect_numeric_features(gdf):
    """
    Detect numeric columns suitable for statistical visualization.
    """
    return [c for c in gdf.columns if gdf[c].dtype in ['float64', 'int64']]


# ============================================================
# 2️⃣ Generate Statistical Summary
# ============================================================
def generate_stat_summary(gdf, columns=None):
    """
    Generate descriptive statistics for given columns in a GeoDataFrame.
    """
    if gdf is None or gdf.empty:
        print("⚠️ Empty GeoDataFrame provided.")
        return None

    if columns is None:
        columns = detect_numeric_features(gdf)

    summary = gdf[columns].describe().T
    summary["missing_values"] = gdf[columns].isna().sum()
    return summary.round(3)


# ============================================================
# 3️⃣ Histogram Plot
# ============================================================
def create_histogram(gdf, column, title="Feature Distribution"):
    """
    Creates a histogram for a selected feature column.
    """
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in GeoDataFrame.")

    fig = px.histogram(
        gdf,
        x=column,
        nbins=30,
        title=f"{title}: {column}",
        color_discrete_sequence=['teal']
    )
    fig.update_layout(
        template="plotly_white",
        bargap=0.1,
        xaxis_title=column,
        yaxis_title="Count"
    )
    return fig


# ============================================================
# 4️⃣ Scatter Matrix Plot
# ============================================================
def create_scatter_matrix(gdf, columns, title="Feature Correlations"):
    """
    Creates an interactive scatter matrix for multiple features.
    """
    if not columns:
        columns = detect_numeric_features(gdf)

    fig = px.scatter_matrix(
        gdf,
        dimensions=columns[:6],  # Limit to first 6 for clarity
        title=title,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig.update_layout(
        template="plotly_white",
        height=700,
        width=900
    )
    return fig


# ============================================================
# 5️⃣ Moran’s I Heatmap (Placeholder — for later integration)
# ============================================================
def create_moran_heatmap(df_results, title="Moran's I (Before vs After Scaling)"):
    """
    Creates a Moran’s I heatmap visualization for SCC diagnostics.
    """
    if df_results is None or df_results.empty:
        print("⚠️ Empty DataFrame provided for Moran’s I heatmap.")
        return None

    fig = px.imshow(
        df_results[['Moran_I_Before', 'Moran_I_After', 'Δ_I']].T,
        labels=dict(x="Feature", y="Moran’s I Metrics", color="Value"),
        x=df_results['Feature'],
        title=title,
        color_continuous_scale="RdYlBu_r"
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        width=900
    )
    return fig

# ============================================================
# 6️⃣ Integration Function for Dashboard
# ============================================================
def generate_visualizations(gdf):
    """
    Returns key plots (histogram + scatter matrix) for dashboard integration.
    """
    if gdf is None or gdf.empty:
        return {}

    numeric_cols = detect_numeric_features(gdf)
    summary = generate_stat_summary(gdf, numeric_cols)
    hist_fig = create_histogram(gdf, numeric_cols[0])
    scatter_fig = create_scatter_matrix(gdf, numeric_cols[:4])

    return {"histogram": hist_fig,"scatter": scatter_fig}
