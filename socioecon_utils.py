import geopandas as gpd
import os

BASE_SOCIO_PATH = r"C:\Users\Oladayiye A S\PHD_Files\NPC\analysis_reports\gdis_data\input_socioecon_files"

def load_socioeconomic_data(lga_name):
    folder = os.path.join(BASE_SOCIO_PATH, lga_name.lower())
    socio_files = [f for f in os.listdir(folder) if f.endswith('.gpkg') or f.endswith('.shp')]
    
    dfs = []
    for f in socio_files:
        path = os.path.join(folder, f)
        gdf = gpd.read_file(path)
        dfs.append(gdf)
    
    # Merge on spatial join or a unique key (e.g., grid_id)
    socio_gdf = dfs[0]
    for other in dfs[1:]:
        socio_gdf = socio_gdf.merge(other, on='grid_id', how='left')
    return socio_gdf


def merge_with_clusters(cluster_gdf, lga_name):
    socio_gdf = load_socioeconomic_data(lga_name)
    merged = cluster_gdf.merge(socio_gdf, on='grid_id', how='left')
    return merged
