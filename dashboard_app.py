# ==============================================================
# 📊 Spatial Clustering Dashboard (Eti-Osa & Surulere)
# ==============================================================
import os
import time
import threading
import traceback
import warnings
from shapely.geometry import shape
import geopandas as gpd
import pandas as pd
import numpy as np
import panel as pn
import plotly.express as px
import plotly.figure_factory as ff
import folium
from spopt.region import WardSpatial, Skater, AZP
from libpysal import weights
from sklearn.preprocessing import MinMaxScaler
from branca.colormap import linear
from folium.features import GeoJsonPopup, GeoJsonTooltip
from socioecon_utils import merge_with_clusters
from config import MODEL_DIRS, SOCIOECON_DIRS, OTHER_LAYERS_DIR
from visualization_utils import create_cluster_stats_plot, create_dendrogram
import tempfile

warnings.filterwarnings("ignore")
pn.extension('plotly', 'floatpanel', 'tabulator')

map_panel = pn.pane.HTML(sizing_mode="stretch_both", height=700)
stats_panel = pn.pane.Plotly(height=500, sizing_mode="stretch_width")
status_text = pn.pane.Markdown("Ready.", sizing_mode="stretch_width")
progress_bar = pn.indicators.Progress(value=0, bar_color="primary", width=400)

# Reactive parameter for selected cluster
selected_cluster = pn.widgets.IntSlider(name="Selected Cluster", start=0, end=10, value=0, step=1)

# --------------------------------------------------------------
# 1️⃣ Dashboard Configuration and Widgets
# --------------------------------------------------------------
lga_selector = pn.widgets.Select(name="Select LGA",options=["EtiOsa", "Surulere"],value="EtiOsa")
grid_selector = pn.widgets.Select(name="Select Grid Level",options=["Grid Level 1", "Grid Level 2"],value="Grid Level 1")
model_selector = pn.widgets.Select(name="Select SCC Model",options=["AZP", "SCHC (Ward Hierarchical)", "SKATER"],value="AZP")

# Cluster selection dropdown (populated dynamically after SCC run)
cluster_selector = pn.widgets.Select(name="Select Cluster",options=["All Clusters"],value="All Clusters",width=250)
view_stats_button = pn.widgets.Button(name="📊 View Cluster Stats",button_type="primary",width=200)
chart_type_selector = pn.widgets.Select(name="Chart Type",options=["Boxplot", "Histogram"],value="Boxplot",width=150)
compare_mode_selector = pn.widgets.Checkbox(name="Compare All Clusters", value=False)

load_button = pn.widgets.Button(name="Load Data", button_type="primary")
run_scc_button = pn.widgets.Button(name="Run SCC Model", button_type="primary",width=150)

status_text = pn.pane.Markdown("🛰 Ready to begin...", sizing_mode="stretch_width")
map_panel = pn.pane.HTML(sizing_mode="stretch_both", height=700)

# =========================================================
# 🔗 Event Bindings: connect buttons to actions
# =========================================================

# Load Data button: just loads and displays map
load_button.on_click(
    lambda event: load_and_render(lga_selector.value, grid_selector.value)
)

# Run SCC Model button: triggers clustering (currently reuses load_and_render)
run_scc_button.on_click(
    lambda event: threading.Thread(
        target=run_scc_background,
        args=(lga_selector.value, grid_selector.value, model_selector.value)
    ).start()
)

# Dashboard Controls
controls = pn.Row(
    lga_selector,
    grid_selector,
    model_selector,
    load_button,  
    pn.Spacer(width=10),
    run_scc_button,
    sizing_mode="stretch_width"
)
# Progress and status indicators
progress_bar = pn.widgets.Progress(name='Clustering Progress', value=0, width=400, bar_color='primary')
# Full Dashboard Layout
layout = pn.Column(
    pn.pane.Markdown("## 🗺️ Spatially Constrained Clustering (SCC) Dashboard"),
    controls,
    status_text,
    progress_bar,
    map_panel,
    stats_panel,
    sizing_mode="stretch_width"
)
# ==============================================================
# 2️⃣ Helper Functions
# ==============================================================

def create_cluster_map(gdf, cluster_col, lga_name, model_name):
    """Create a Folium map showing cluster zones with legends and zoom."""
    centroid = gdf.geometry.unary_union.centroid
   
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="CartoDB positron")

    folium.GeoJson(
        gdf,
        name=f"{model_name} Clusters",
        style_function=lambda x: {
            "fillColor": f"#{np.random.randint(0, 0xFFFFFF):06x}",
            "color": "black",
            "weight": 0.8,
            "fillOpacity": 0.6
        },
        tooltip=folium.GeoJsonTooltip(fields=[cluster_col], aliases=["Cluster"])
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ---------------------------------------------------------------------
# ==============================================================
# Master function: run_clustering
# ==============================================================
def run_clustering(lga_name, grid_level, method="AZP", progress_callback=None, save_output=True, pre_scaled=False):
    """
    Unified SCC clustering function with metrics reporting.
    - Reads LGA-specific .gpkg data.
    - Selects morphometric features.
    - Cleans, scales, and clusters data.
    - Computes Silhouette score and pseudo R².
    - Optionally saves clustered output.
    """
    try:
        # --- STEP 1: File setup and loading ---
        base_dir = r"C:\Users\Oladayiye A S\PHD_Files\NPC\analysis_reports\gdis_data\model_files"
        lga_folder = os.path.join(base_dir, lga_name.lower())

        if not os.path.exists(lga_folder):
            raise FileNotFoundError(f"❌ LGA folder not found: {lga_folder}")

        # Extract grid number (e.g., "1" or "2")
        grid_suffix = str(grid_level).replace("Grid Level", "").strip()[-1]

        # Define common filename patterns
        expected_files = [
            f"{lga_name}_L{grid_suffix}_selected.gpkg",
            f"{lga_name}_{grid_suffix}_selected.gpkg",
            f"{lga_name}_GridLevel{grid_suffix}_selected.gpkg",
            f"{lga_name}_grid{grid_suffix}_selected.gpkg"
        ]

        # Look for a match
        available_files = [f for f in os.listdir(lga_folder) if f.endswith(".gpkg")]
        matched_file = None
        for pattern in expected_files:
            for f in available_files:
                if pattern.lower() in f.lower():
                    matched_file = f
                    break
            if matched_file:
                break

        if not matched_file:
            raise FileNotFoundError(
                f"❌ Could not find any .gpkg file for {lga_name} ({grid_level}).\n"
                f"Tried patterns: {expected_files}\n"
                f"Available files: {available_files}"
            )

        input_file = os.path.join(lga_folder, matched_file)
        print(f"✅ Using detected dataset: {input_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"❌ Input file not found: {input_file}")

        if progress_callback:
            progress_callback(5)

        gdf = gpd.read_file(input_file)
        gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)

        # --- STEP 2: Compute spatial weights ---
        w = weights.Queen.from_dataframe(gdf)
        w.transform = "r"

        if progress_callback:
            progress_callback(20)

        # --- STEP 3: Feature selection ---
        morpho_vars = [
            'building_c', 'area_mean', 'area_sd', 'area_cv',
            'compactnes', 'nndist_mea', 'nndist_cv',
            'angle_mean', 'angle_vari', 'angle_entr',
            'built_up_r', 'green_spac'
        ]
        feature_cols = [c for c in morpho_vars if c in gdf.columns]
        if len(feature_cols) < 3:
            print(f"[WARNING] Insufficient morpho_vars found for {lga_name} ({grid_level}). Using numeric columns instead.")
            feature_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()

        X = gdf[feature_cols].select_dtypes(include=[np.number])

        # --- STEP 4: Clean invalid or missing values ---
        X = X.replace([np.inf, -np.inf], np.nan)
        invalid_counts = X.isnull().sum()
        if invalid_counts.any():
            print(f"[WARNING] Missing or invalid numeric data in {lga_name} ({grid_level}):")
            print(invalid_counts[invalid_counts > 0])
            X = X.fillna(X.mean())

        constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
        if constant_cols:
            print(f"[INFO] Dropping constant features: {constant_cols}")
            X = X.drop(columns=constant_cols)

        print(f"[DEBUG] Cleaned dataset for {lga_name} ({grid_level}) — shape: {X.shape}")
        print(f"[DEBUG] Columns ready for clustering: {X.columns.tolist()}")

        # 🧩 Debug missing values before final cleaning
        print("[DEBUG] Missing values per column before cleaning:")
        print(X.isnull().sum()[X.isnull().sum() > 0])

        # 🧹 Final robust cleaning before scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        X = X.dropna(axis=1, how='all')   # Drop empty columns
        X = X.dropna(axis=0, how='any')   # Drop rows with NaN

        zero_var_cols = [c for c in X.columns if X[c].nunique() <= 1]
        if zero_var_cols:
            print(f"[INFO] Dropping zero-variance columns: {zero_var_cols}")
            X = X.drop(columns=zero_var_cols)

        # ✅ Final safeguard before scaling
        # Drop rows where all columns are NaN
        X = X.dropna(axis=0, how='all')

        # Drop any remaining rows with NaNs
        if X.isnull().values.any():
            print("[WARNING] Final cleanup: removing remaining NaN rows before scaling...")
            nan_rows = X[X.isnull().any(axis=1)]
            print(f"[INFO] Removing {len(nan_rows)} problematic rows.")
            X = X.dropna(axis=0, how='any')

        # Check again and fill any edge NaNs (if none dropped)
        if X.isnull().values.any():
            print("[WARNING] Remaining NaN values after cleaning — replacing with zeros.")
            X = X.fillna(0)
        
        print(f"[DEBUG] Final feature matrix shape for {lga_name} ({grid_level}): {X.shape}")
       
        # --- STEP X: Sync cleaned data back into GeoDataFrame ---
        for col in X.columns:
            gdf[col] = X[col].values
            
        print(f"[INFO] Cleaned numeric features reattached to gdf for {lga_name} ({grid_level}).")  
        # --- STEP X+1: Validate data integrity before clustering ---
        nan_check = gdf[feature_cols].isnull().sum()
        if nan_check.any():
            print("\n[❌ ERROR] Some features still contain NaN values before clustering:")
            print(nan_check[nan_check > 0])
            raise ValueError(f"NaN values remain in {len(nan_check[nan_check > 0])} feature(s). Please inspect data cleaning for {lga_name} ({grid_level}).")
        else:
            print(f"[✅] No NaN values detected in clustering features for {lga_name} ({grid_level}).")

        # =============================================================
        # STEP 4B: Handle scaling depending on data source
        # =============================================================

        if pre_scaled:
            print(f"[INFO] {lga_name} ({grid_level}) data is pre-scaled — skipping normalization.")
            X_scaled = X.values  # use as-is
        else:
            print(f"[INFO] Performing MinMax scaling for {lga_name} ({grid_level})...")
            X_scaled = MinMaxScaler().fit_transform(X)

        # --- Sync cleaned (or pre-scaled) data back into gdf ---
        for col in X.columns:
            gdf[col] = X[col].values

        print(f"[✅] Feature matrix ready — shape: {X.shape}, scaled={not pre_scaled}")
        
        # --- STEP 5A: Choose and run clustering ---
        method = method.upper().strip()
        method = (
            "SCHC" if "SCHC" in method or "WARD" in method
            else "AZP" if "AZP" in method
            else "SKATER" if "SKATER" in method
            else method
        )
        n_clusters = {"AZP": 6, "SKATER": 9, "SCHC": 8}.get(method, 6)

        print(f"🔄 Running {method} clustering for {lga_name} ({grid_level}) using {len(feature_cols)} features...")

        if method == "AZP":
            model = AZP(gdf, w, attrs_name=feature_cols, n_clusters=n_clusters)
        elif method == "SKATER":
            model = Skater(gdf, w, attrs_name=feature_cols, n_clusters=n_clusters)
        elif method == "SCHC":
            model = WardSpatial(gdf, w, attrs_name=feature_cols, n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported method: {method}")

        model.solve()
        gdf["cluster"] = model.labels_

        if progress_callback:
            progress_callback(85)

        # --- STEP 6: Compute metrics ---
        metrics = {}
        try:
            metrics["Silhouette"] = silhouette_score(X_scaled, model.labels_) if len(set(model.labels_)) > 1 else np.nan
            cluster_means = np.array([X_scaled[np.array(model.labels_) == k].mean(axis=0) for k in np.unique(model.labels_)])
            global_mean = X_scaled.mean(axis=0)
            ss_total = np.sum((X_scaled - global_mean) ** 2)
            ss_between = np.sum([len(X_scaled[np.array(model.labels_) == k]) * np.sum((m - global_mean) ** 2)
                                 for k, m in zip(np.unique(model.labels_), cluster_means)])
            metrics["R²"] = ss_between / ss_total if ss_total > 0 else np.nan
        except Exception as metric_err:
            print(f"[WARNING] Metric calculation failed: {metric_err}")
            metrics = {"R²": np.nan, "Silhouette": np.nan}

        # --- STEP 7: Save output ---
        if "cluster" not in gdf.columns:
            raise ValueError(f"Clustering failed — no 'cluster' column generated for {lga_name} ({grid_level}).")

        if save_output:
            output_file = os.path.join(base_dir, lga_name.lower(), f"{lga_name}_{grid_level[-1]}_{method}_clusters.gpkg")
            gdf.to_file(output_file, driver="GPKG")
            print(f"💾 Saved clustered output: {output_file}")

        if progress_callback:
            progress_callback(100)

        print(f"✅ {method} clustering complete for {lga_name} ({grid_level}) using {len(feature_cols)} features.")
        print(f"📈 Metrics → R²: {metrics['R²']:.3f}, Silhouette: {metrics['Silhouette']:.3f}")

        gdf.attrs["metrics"] = metrics
        gdf.attrs["features_used"] = feature_cols

        return gdf

    except Exception as e:
        print(f"[ERROR] run_clustering failed for {lga_name} ({grid_level}): {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================
# CLI test (for standalone use)
# ==============================================================
if __name__ == "__main__":
    print("Running clustering engine test...")
    result = run_clustering("Eti-Osa", grid_level="L2", method="SCHC", save_output=True)
    print(result.head())

# ---------------------------------------------------------------------
# 🧬 Dendrogram Generation & Map Synchronization
# ---------------------------------------------------------------------
def generate_dendrogram(gdf, model_name, linkage_matrix):
    """Create interactive dendrogram for SCHC or hierarchical clustering."""
    dendro_fig = ff.create_dendrogram(
        linkage_matrix,
        orientation="bottom",
        color_threshold=0.7 * np.max(linkage_matrix[:, 2])
    )
    dendro_fig.update_layout(
        width=1000,
        height=400,
        title=f"{model_name} Dendrogram",
        showlegend=False,
        template="simple_white"
    )
    # Wrap in a Panel Plotly pane
    return pn.pane.Plotly(dendro_fig, height=400, sizing_mode="stretch_width")

def show_dendrogram_modal(gdf, model_name, linkage_matrix):
    """
    Display the dendrogram in a floating modal window.
    """
    dendro_pane = generate_dendrogram(gdf, model_name, linkage_matrix)

    modal = pn.layout.FloatPanel(
        dendro_pane,
        name="SCHC Dendrogram Viewer",
        position="center",
        width=1100,
        height=500,
        sizing_mode="stretch_both"
    )

    modal.open = True  # Automatically open when created
    return modal

def bind_dendrogram_map(fig_widget, gdf, map_panel):
    """Interactive linkage between dendrogram and map display."""
    def on_dendro_click(event):
        # When a branch is clicked, highlight corresponding clusters on map
        if event.new and "points" in event.new:
            clicked_cluster = int(event.new["points"][0]["pointIndex"])
            selected_rows = gdf[gdf["cluster"] == clicked_cluster]
            map_panel.object = selected_rows.explore(color="red")

    fig_widget.param.watch(on_dendro_click, "click_data")

# ==============================================================
# 3️⃣ Core Function — Load and Render Map & Model
# ==============================================================
def load_and_render(lga_name, grid_level):
    """
    Load and render the selected dataset dynamically from config.py.
    Shows clear dashboard feedback for errors and successes.
    """
    if not lga_name or not grid_level:
        raise ValueError(f"Missing argument(s): lga_name={lga_name}, grid_level={grid_level}")
    try:
        key = lga_name.lower().replace("-", "").strip()

        # --- STEP 1: Validate LGA name ---
        if key not in MODEL_DIRS:
            available = ", ".join(MODEL_DIRS.keys())
            msg = f"⚠️ No model directory found for '{lga_name}'. Available: [{available}]"
            status_text.object = f"<span style='color:orange;'>{msg}</span>"
            pn.state.notifications.warning(msg)
            return None

        # --- STEP 2: Build correct model file path ---
        base_path = r"C:\Users\Oladayiye A S\PHD_Files\NPC\analysis_reports\gdis_data\model_files"

        lga_clean = lga_name.replace("-", "").replace(" ", "")
        grid_num = grid_level.split()[-1]
        file_key = f"{lga_clean}_L{grid_num}_selected.gpkg"

        model_path = os.path.join(base_path, lga_name.lower(), file_key)

        # --- STEP 3: Check file existence ---
        if not os.path.exists(model_path):
            msg = f"❌ No file found at: {model_path}"
            status_text.object = f"<span style='color:red;'>{msg}</span>"
            pn.state.notifications.error(msg)
            return None

        # --- STEP 4: Load the dataset ---
        gdf = gpd.read_file(model_path)

        if gdf is None or gdf.empty:
            msg = f"⚠️ Dataset at {model_path} is empty or could not be loaded."
            status_text.object = f"<span style='color:orange;'>{msg}</span>"
            pn.state.notifications.warning(msg)
            return None

        # --- STEP 5: Report success ---
        msg = f"✅ Loaded {lga_name} {grid_level} successfully – {len(gdf)} polygons found."
        status_text.object = f"<span style='color:green;'>{msg}</span>"
        
        global loaded_gdf
        loaded_gdf = gdf
        return gdf

    except Exception as e:
        # --- STEP 6: Handle unexpected errors ---
        msg = f"❌ Error loading {lga_name} {grid_level}: {e}"
        status_text.object = f"<span style='color:red;'>{msg}</span>"
        pn.state.notifications.error(msg)
        traceback.print_exc()
        return None
    
def on_load_click(event):
        lga_name = lga_selector.value
        grid_level = grid_selector.value
        print(f"🔹 Load button clicked for {lga_name} ({grid_level})")
        load_and_render(lga_name, grid_level)

# =====================================================
# 🗺️ Cluster Map Generator — Reusable Folium Component
# =====================================================
def create_cluster_map(gdf, cluster_col, lga_name, model_name):
    """
    Create an interactive Folium map showing cluster zones with legends, zoom, and tooltips.
    """
    gdf = gdf.to_crs(epsg=4326)
    centroid = gdf.geometry.unary_union.centroid

    # Initialize map
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="CartoDB positron")

    # Generate color palette based on number of clusters
    num_clusters = gdf[cluster_col].nunique()
    color_scale = linear.Set3_09.scale(0, num_clusters - 1)

    # Style function for polygons
    def style_function(feature):
        cluster_id = feature["properties"][cluster_col]
        return {
            "fillColor": color_scale(cluster_id),
            "color": "black",
            "weight": 0.7,
            "fillOpacity": 0.6
        }

    # Add GeoJson with tooltip and click highlighting
    geojson = folium.GeoJson(
        gdf,
        name=f"{model_name} Clusters",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[cluster_col], aliases=["Cluster:"]),
        highlight_function=lambda f: {"color": "red", "weight": 3}
    )
    geojson.add_to(m)

    # Add color legend and layer control
    color_scale.caption = f"{model_name} Cluster Groups ({lga_name})"
    color_scale.add_to(m)
    folium.LayerControl().add_to(m)

    return m
# =============================================================
#  Create the asynchronous SCC runner
# ==============================================================
def run_scc_background(lga_name, grid_level, method):
    """
    Background thread: runs SCC clustering and updates dashboard panels.
    """
    clustered_gdf = None
    try:
        status_text.value = f"🏃 Running {method} clustering for {lga_name} ({grid_level})..."
        progress_bar.value = 0
        print(f"[INFO] Running {method} clustering for {lga_name} ({grid_level})...")

        # Run clustering and update progress bar
        clustered_gdf = run_clustering(
            lga_name, grid_level, method,
            progress_callback=lambda p: setattr(progress_bar, 'value', p)
        )

        if clustered_gdf is None or clustered_gdf.empty:
            status_text.value = f"⚠️ Clustering failed for {lga_name} ({grid_level}) — please check your data or selected method."
            progress_bar.value = 0
            return

        # 🗺️ Render cluster map
        status_text.value = "🗺️ Rendering clustered map..."
        print("[INFO] Rendering clustered map...")

        m = create_cluster_map(clustered_gdf, "cluster", lga_name, method)
        map_panel.object = m._repr_html_()

        # 📊 Generate cluster summary chart
        status_text.value = "📊 Generating cluster statistics..."
        stats_fig = px.histogram(
            clustered_gdf,
            x="cluster",
            color="cluster",
            title=f"Cluster Size Distribution for {lga_name} ({grid_level})",
        )
        stats_panel.object = pn.pane.Plotly(stats_fig, height=400, sizing_mode="stretch_width")

        progress_bar.value = 100
        print(f"[SUCCESS] {method} clustering complete for {lga_name} ({grid_level}).")

        # Display metrics in dashboard footer
        metrics = clustered_gdf.attrs.get("metrics",{})
        if metrics:
            metrics_text = f"📈 R²: {metrics.get('R²', np.nan):.3f} | Silhouette: {metrics.get('Silhouette', np.nan):.3f}"
            status_text.value  += f"<br>{metrics_text}"

        # 🧬 SCHC (Hierarchical) Dendrogram Modal
        if "SCHC" in method.upper():

            print("[INFO] Generating SCHC dendrogram modal...")

            linkage_matrix = np.column_stack([
                np.arange(len(clustered_gdf) - 1),
                np.arange(1, len(clustered_gdf)),
                np.random.random(len(clustered_gdf) - 1),
                np.ones(len(clustered_gdf) - 1)
            ])

            dendrogram_modal = show_dendrogram_modal(clustered_gdf, method, linkage_matrix)
            pn.state.add_panel(dendrogram_modal)  # ✅ replaces layout.append()
            status_text.value = f"✅ SCHC dendrogram displayed for {lga_name} ({grid_level})."

    except Exception as e:
        status_text.value = f"❌ Error during SCC run: {str(e)}"
        print(f"[ERROR] {e}", flush=True)
        
        # 🟢 Non-SCHC Models (AZP, SKATER, etc.)
        progress_bar.value = 80
        status_text.object = "🗺️ Rendering clustered map..."

        # Generate map visualization
        m = create_cluster_map(clustered_gdf, "cluster", lga_name, method)
        map_panel.object = m._repr_html_()

        # Generate cluster statistics visualization
        stats_fig = px.histogram(
            clustered_gdf,
            x="cluster",
            color="cluster",
            title=f"Cluster Size Distribution for {lga_name} ({grid_level})"
        )
        stats_panel.object = pn.pane.Plotly(stats_fig, height=400, sizing_mode="stretch_width")

        progress_bar.value = 100
        status_text.value = f"✅ Clustering complete for {lga_name} ({grid_level})."
        
    except Exception as e:
        status_text.value = f"❌ Error during SCC run: {str(e)}"
        print(f"[ERROR] {e}", flush=True)

# ======================================================================
#  Full render Map Function - displays the SCC clusters interactively in a zoomable map,
#  Assigns unique colours for each cluster, include a dynamic legend, and support click ppoups
#  showing cluster attributes
# =====================================================================
def render_map(clustered_gdf):
    """
    Render SCC cluster results on an interactive folium map.
    Adds legends, color schemes, zoom, and interactivity.
    """
    global stats_panel
    if clustered_gdf is None or "geometry" not in clustered_gdf.columns:
        status_text.object = "⚠️ No spatial data available to render."
        return

    # --- Step 1: Clean and prepare ---
    clustered_gdf = clustered_gdf.to_crs(epsg=4326)
    clustered_gdf["cluster"] = clustered_gdf["cluster"].astype(str)

    # --- Step 2: Update cluster selector dynamically ---
    unique_clusters = sorted(clustered_gdf["cluster"].unique())
    cluster_selector.options = ["All Clusters"] + [str(c) for c in unique_clusters]
    cluster_selector.value = "All Clusters"

    # --- Step 3: Define color map ---
    unique_clusters = clustered_gdf["cluster"].unique()
    colormap = linear.Set1_09.scale(0, len(unique_clusters))
    cluster_colors = {str(c): colormap(i) for i, c in enumerate(unique_clusters)}

    # --- Step 4: Create folium map centered on extent ---
    centroid = clustered_gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")

    # --- Step 5: Define tooltip and popup info ---
    tooltip_fields = [c for c in clustered_gdf.columns if c not in ["geometry"]]
    popup = GeoJsonPopup(
        fields=tooltip_fields,
        aliases=[f"{c}: " for c in tooltip_fields],
        localize=True,
        labels=True,
        style="background-color: white;",
    )

    tooltip = GeoJsonTooltip(fields=["cluster"], aliases=["Cluster: "])

    # --- Step 5: Add GeoJSON layer ---
    folium.GeoJson(
        clustered_gdf,
        style_function=lambda feature: {
            "fillColor": cluster_colors[str(feature["properties"]["cluster"])],
            "color": "black",
            "weight": 0.6,
            "fillOpacity": 0.7,
        },
        highlight_function=lambda x: {"weight": 3, "color": "blue"},
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    # --- Step 6: Add legend ---
    colormap.caption = "Cluster ID"
    colormap.add_to(m)

    # --- Step 7: Add graticule / grid overlay (optional aesthetic touch) ---
    try:
        import folium.plugins as plugins
        plugins.MousePosition().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MeasureControl().add_to(m)
    except Exception as e:
        print("Optional folium plugins not available:", e)

    # --- Step 8: Save map to temporary HTML and display in dashboard ---
    tmp_dir = tempfile.gettempdir()
    map_path = os.path.join(tmp_dir, "cluster_map.html")
    m.save(map_path)

    map_panel.object = f'<iframe src="{map_path}" width="100%" height="700"></iframe>'

    # Panel to display statistical visualization
    stats_panel = pn.pane.Plotly(height=400, sizing_mode="stretch_width")

def filter_and_render(event=None):
    """
    Filters the current clustered GeoDataFrame by the selected cluster
    and re-renders the map interactively.
    """
    global last_clustered_gdf

    if "last_clustered_gdf" not in globals() or last_clustered_gdf is None:
        status_text.object = "⚠️ No clustering results available yet."
        return

    if cluster_selector.value == "All Clusters":
        filtered_gdf = last_clustered_gdf
        status_text.object = "🗺️ Showing all clusters."
    else:
        filtered_gdf = last_clustered_gdf[last_clustered_gdf["cluster"] == cluster_selector.value]
        status_text.object = f"🔎 Showing only cluster {cluster_selector.value}."

    render_map(filtered_gdf)
    # Bind the dropdown interaction such that dropdown selection is changed, map automatically updates
    cluster_selector.param.watch(filter_and_render, 'value')
# ============================================================
# Chart Type Selector and Comparisons
# ============================================================
def show_cluster_stats(event=None):
    """
    Generates feature-level distributions for the selected cluster,
    with optional comparison across all clusters.
    Supports Boxplot and Histogram chart types.
    """
    global last_clustered_gdf

    if "last_clustered_gdf" not in globals() or last_clustered_gdf is None:
        status_text.object = "⚠️ Please run an SCC model first."
        return

    # Check cluster selection
    if cluster_selector.value == "All Clusters" and not compare_mode_selector.value:
        status_text.object = "⚠️ Please select a specific cluster or enable comparison mode."
        return

    # --- Data preparation ---
    if compare_mode_selector.value:
        # Compare across all clusters
        plot_df = last_clustered_gdf.melt(
            id_vars=["cluster"],
            var_name="Feature",
            value_name="Value"
        )
    else:
        # Single cluster visualization
        selected_cluster = cluster_selector.value
        cluster_gdf = last_clustered_gdf[last_clustered_gdf["cluster"] == selected_cluster]
        plot_df = cluster_gdf.melt(var_name="Feature", value_name="Value")
        plot_df["cluster"] = selected_cluster

    # --- Chart creation ---
    if compare_mode_selector.value:
        # Multi-cluster visualization
        if chart_type_selector.value == "Boxplot":
            fig = px.box(
                plot_df,
                x="Feature",
                y="Value",
                color="cluster",
                color_discrete_sequence=px.colors.qualitative.Set1,
                title="Comparison Across All Clusters",
                template="plotly_white"
            )
        else:
            fig = px.histogram(
                plot_df,
                x="Value",
                color="cluster",
                facet_col="Feature",
                facet_col_wrap=3,
                color_discrete_sequence=px.colors.qualitative.Set1,
                title="Histogram Comparison Across Clusters",
                template="plotly_white"
            )
    else:
        # Single-cluster visualization
        selected_cluster = cluster_selector.value
        if chart_type_selector.value == "Boxplot":
            fig = px.box(
                plot_df,
                x="Feature",
                y="Value",
                points="all",
                color_discrete_sequence=["#1f77b4"],
                title=f"Distribution of Attributes — Cluster {selected_cluster}",
                template="plotly_white"
            )
        else:
            fig = px.histogram(
                plot_df,
                x="Value",
                color="Feature",
                marginal="box",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                title=f"Histogram — Cluster {selected_cluster}",
                template="plotly_white"
            )

    # --- Chart layout and display ---
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Value",
        title_font=dict(size=14, family="Arial", color="black"),
        margin=dict(l=40, r=40, t=50, b=30),
        legend_title_text="Cluster"
    )

    stats_panel.object = fig

    if compare_mode_selector.value:
        status_text.object = "✅ Displaying comparison across all clusters."
    else:
        status_text.object = f"✅ Displaying attribute distributions for Cluster {selected_cluster}"
# ====================================================
# 🔗 Event Bindings (connect buttons to functions)
# ====================================================
# Bind the load button to the dataset loader
load_button.on_click(lambda event: load_and_render(lga_selector.value,grid_selector.value))

# ==============================================================
# 4️⃣ Dashboard Layout
# ==============================================================

controls = pn.Row(
    lga_selector,
    grid_selector,
    model_selector,
    load_button,
    run_scc_button,
    sizing_mode="stretch_width"
)
# =========================================================
#  🔗 Event Bindings (connect buttons to functions)
# =========================================================
# 1️⃣ Bind the "Load Data" button
def on_load_data(event):
    """Triggered when the user clicks 'Load Data'."""
    lga_name = lga_selector.value
    grid_level = grid_selector.value
    status_text.object = f"🔄 Loading data for **{lga_name}** ({grid_level})..."
    try:
        #gdf = load_and_render(lga_name, grid_level)
        if gdf is not None:
            status_text.object = f"✅ Loaded **{lga_name}** ({grid_level}) successfully! ({len(gdf)} records)"
        else:
            status_text.object = f"⚠️ No data returned for **{lga_name}** ({grid_level})."
    except Exception as e:
        status_text.object = f"❌ Error loading data: {str(e)}"
        print(e)
    # Bind the "Run SCC Model" button to the clustering function
    run_scc_button.on_click(
    lambda event: threading.Thread(
        target=run_scc_background,
        args=(lga_selector.value, grid_selector.value, model_selector.value),
        daemon=True # ensures thread stops when app close
    ).start()
)
# load_button.on_click(on_load_data)
# 2️⃣ Bind the "Run SCC Model" button
def on_run_scc(event):
    """Triggered when the user clicks 'Run SCC Model'."""
    lga_name = lga_selector.value
    grid_level = grid_selector.value
    model_name = model_selector.value
        # ✅ Reset status and progress
    status_text.value = "⏳ Preparing to run model..."
    progress_bar.value = 0   
    thread = threading.Thread(
        target=run_scc_background,
        args=(lga_name, grid_level, model_name)
    )
    thread.start()
run_scc_button.on_click(on_run_scc)

# ===============================================================
#  Call the SCC Model
# ===============================================================
def start_scc_run(event=None):
    lga_name = lga_selector.value
    grid_level = grid_selector.value.split()[-1]  # Extract numeric level
    model = model_selector.value
    selected_gdf = load_and_render(lga_name, grid_level)  # Your dataset loader

    status_text.object = f"🧮 Initializing {model} for {lga_name} Grid {grid_level}..."
    progress_bar.value = 0

    # Launch in background thread
    thread = threading.Thread(target=run_scc_background, args=(lga_name, grid_level, model, selected_gdf))
    thread.start()

# =========================================================
# 🚀 Launch Dashboard (Development & Deployment Compatible)
# =========================================================

import os
import panel as pn

# Detect whether running in hosted mode (e.g., Hugging Face, Panel Cloud)
# or in local development mode
IS_DEV = os.environ.get("PANEL_DEV_MODE", "0") == "1"

if IS_DEV or __name__ == "__main__":
    print("🔧 Running in local development mode...")
    pn.serve(
        layout,
        title="Spatial Clustering Dashboard",
        show=True,
        port=5006,
        autoreload=True,
    )
else:
    # Running on server (e.g., 'panel serve dashboard_app.py')
    print("🚀 Running in production/hosted mode...")
    layout.servable()