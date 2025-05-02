import geopandas as gpd
import pandas as pd

def load_triplegs_geojson(filepath, parse_dates=None, crs="EPSG:4326"):
    """
    Load a GeoJSON file containing triplegs into a GeoDataFrame.

    Parameters:
    - filepath (str): Path to the GeoJSON file.
    - parse_dates (list): List of timestamp column names to parse (optional).
    - crs (str): Coordinate Reference System. Default is 'EPSG:4326' (WGS84).

    Returns:
    - GeoDataFrame with parsed geometry and timestamps.
    """
    # Load using GeoPandas
    gdf = gpd.read_file(filepath)

    # Parse datetime columns if specified
    if parse_dates:
        for col in parse_dates:
            if col in gdf.columns:
                gdf[col] = pd.to_datetime(gdf[col], errors='coerce')

    # Ensure CRS is set (especially if missing in file)
    if gdf.crs is None:
        gdf.set_crs(crs, inplace=True)

    return gdf

def load_districts_layer(
    filepath: str,
    layer: str = None,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Load a spatial layer of municipalities, parishes, or districts.

    Parameters:
    - filepath (str): Path to spatial dataset (e.g. .shp, .gpkg, or folder for .gpkg with multiple layers)
    - layer (str): Layer name (only needed for GeoPackage or multi-layer datasets)
    - crs (str): Coordinate Reference System to reproject to (default: EPSG:4326)

    Returns:
    - GeoDataFrame with geometry and spatial attributes
    """
    try:
        if layer:
            gdf = gpd.read_file(filepath, layer=layer)
        else:
            gdf = gpd.read_file(filepath)
    except Exception as e:
        raise IOError(f"Could not read file '{filepath}': {e}")

    if gdf.crs is None:
        print("[Info] No CRS detected in dataset. Assigning default:", crs)
        gdf.set_crs(crs, inplace=True)
    else:
        gdf = gdf.to_crs(crs)

    return gdf