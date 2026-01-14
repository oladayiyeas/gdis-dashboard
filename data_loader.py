"""
data_loader.py
--------------------------------------------
Handles loading of model files, socioeconomic datasets, and
optional GIS layers (GeoJSON / GeoPackage) for integration
into the open-source SCC Dashboard.

Author: [Your Name]
Version: 1.1 (PhD GDIS Workflow)
--------------------------------------------
"""

import os
import glob
import geopandas as gpd
import pandas as pd


# ============================================================
# 1️⃣ — Model File Search and Load
# ============================================================

def search_model_file(model_dir, grid_level):
    """
    Searches for a GeoPackage model file for a given grid level
    within a specified directory.

    Args:
        model_dir (str): Path to folder containing model GeoPackages.
        grid_level (int): Grid level identifier (1 or 2).

    Returns:
        str: Path to the first matching GeoPackage file, or None if not found.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Model directory not found: {model_dir}")

    pattern = f"*L{grid_level}*.gpkg"
    files = glob.glob(os.path.join(model_dir, pattern))

    if not files:
        print(f"⚠️ No model files found in {model_dir} for Grid Level {grid_level}.")
        return None

    print(f"✅ Found model file: {os.path.basename(files[0])}")
    return files[0]


def load_model_data(model_dir, grid_level):
    """
    Loads the model GeoDataFrame for a given grid level.

    Args:
        model_dir (str): Directory containing GeoPackage model files.
        grid_level (int): Grid level identifier (1 or 2).

    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame.
    """
    model_file = search_model_file(model_dir, grid_level)
    if not model_file:
        return None

    try:
        gdf = gpd.read_file(model_file)
        print(f"✅ Model data loaded: {len(gdf)} records from {model_file}")
        return gdf
    except Exception as e:
        print(f"❌ Error loading model data: {e}")
        return None


# ============================================================
# 2️⃣ — Socioeconomic Data Attachment
# ============================================================

def load_and_attach_socioecon(gdf, socioecon_dir, grid_level):
    """
    Loads and attaches a socioeconomic GeoDataFrame to the base model data.

    Args:
        gdf (GeoDataFrame): Base grid GeoDataFrame.
        socioecon_dir (str): Directory containing socioeconomic GeoPackages.
        grid_level (int): Grid level identifier (1 or 2).

    Returns:
        GeoDataFrame: Merged dataset (model + socioeconomic attributes).
    """
    if gdf is None:
        print("⚠️ No base GeoDataFrame provided.")
        return None

    socio_file_pattern = f"*L{grid_level}*_enrchd.gpkg"
    socio_files = glob.glob(os.path.join(socioecon_dir, socio_file_pattern))

    if not socio_files:
        print(f"⚠️ No socioeconomic file found for Grid Level {grid_level}. Returning base gdf.")
        return gdf

    socio_file = socio_files[0]
    print(f"📂 Attaching socioeconomic data: {os.path.basename(socio_file)}")

    try:
        socio_gdf = gpd.read_file(socio_file)
        socio_gdf = socio_gdf.rename(columns=str.lower)
        gdf = gdf.rename(columns=str.lower)

        # Find possible merge keys
        merge_key = next(
            (col for col in ["id", "grid_id", "index", "cell_id"]
             if col in gdf.columns and col in socio_gdf.columns),
            None
        )

        if merge_key:
            merged = gdf.merge(socio_gdf, on=merge_key, how="left")
        else:
            print("⚠️ No common key found — performing spatial join instead.")
            merged = gpd.sjoin(gdf, socio_gdf, how="left", predicate="intersects")

        print(f"✅ Socioeconomic data attached successfully ({len(merged)} records).")
        return merged

    except Exception as e:
        print(f"❌ Error attaching socioeconomic data: {e}")
        return gdf


# ============================================================
# 3️⃣ — Load Other Optional GIS Layers
# ============================================================

def attach_other_layers(other_layers_dir):
    """
    Loads all additional GIS layers (GeoJSON, GeoPackage) in a folder.

    Args:
        other_layers_dir (str): Path to folder containing extra layers.

    Returns:
        list[GeoDataFrame]: List of loaded layers.
    """
    layers = []

    if not os.path.exists(other_layers_dir):
        print(f"⚠️ Directory not found: {other_layers_dir}")
        return layers

    for file in glob.glob(os.path.join(other_layers_dir, "*")):
        if file.endswith((".gpkg", ".geojson", ".json")):
            try:
                layer = gpd.read_file(file)
                layers.append(layer)
                print(f"✅ Loaded additional layer: {os.path.basename(file)} ({len(layer)} features)")
            except Exception as e:
                print(f"⚠️ Failed to load {os.path.basename(file)}: {e}")

    if not layers:
        print("⚠️ No additional layers loaded.")
    else:
        print(f"✅ {len(layers)} additional layers loaded successfully.")
    return layers


# ============================================================
# 4️⃣ — Integrated Data Loading Pipeline
# ============================================================

def load_and_render(lga_name, grid_level, base_dir):
    """
    Unified function to load model, socioeconomic, and other layer data.

    Args:
        lga_name (str): Either 'etiosa' or 'surulere'.
        grid_level (int): Grid level (1 or 2).
        base_dir (str): Base path for gdis_data folder.

    Returns:
        dict: Dictionary containing all loaded data layers.
    """
    lga_name = lga_name.lower()

    model_dir = os.path.join(base_dir, "model_files", lga_name)
    socioecon_dir = os.path.join(base_dir, "input_socioecon_files", lga_name)
    other_layers_dir = os.path.join(base_dir, "other_layers")

    print(f"📂 Loading {lga_name.title()} Grid Level {grid_level} data...")

    gdf = load_model_data(model_dir, grid_level)
    if gdf is None:
        raise ValueError(f"❌ No model data found for {lga_name} Grid Level {grid_level}")

    merged_gdf = load_and_attach_socioecon(gdf, socioecon_dir, grid_level)
    other_layers = attach_other_layers(other_layers_dir)

    return {
        "model_data": merged_gdf,
        "additional_layers": other_layers
    }
