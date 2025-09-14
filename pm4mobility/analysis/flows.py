from shapely.geometry import Point
import geopandas as gpd

def create_od_flows(
    df,
    districts,
    trip_id_col="trip_id",
    geometry_col="geometry",
    mode_col="mode",
    length_col="length",
    duration_col="duration_minutes",
    speed_col="speed",
    started_at_col="started_at",
    finished_at_col="finished_at",
    area_col="Concelho",
    antenna_col="Des_Simpli",
    crs="EPSG:4326"
):
    """
    Create origin-destination (OD) flows from a GeoDataFrame of triplegs.

    Parameters:
    - df: GeoDataFrame with LineString triplegs
    - districts: GeoDataFrame with polygons (must have area_col and antenna_col)
    - crs: Coordinate Reference System to enforce on outputs

    Returns:
    - GeoDataFrame with OD flows (origin, destination, areas, metadata)
    """

    required_cols = [
        trip_id_col, geometry_col, started_at_col, finished_at_col,
        mode_col, length_col, duration_col, speed_col
    ]

    # Validate required columns in triplegs
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in input DataFrame: '{col}'")

    # Check geometry type
    if not isinstance(df, gpd.GeoDataFrame):
        raise TypeError("Input df must be a GeoDataFrame.")

    if not isinstance(districts, gpd.GeoDataFrame):
        raise TypeError("Input districts must be a GeoDataFrame.")

    if df.geometry.iloc[0].geom_type != "LineString":
        raise TypeError("Expected LineString geometries in input df.")

    # Ensure CRS matches
    if df.crs != districts.crs:
        print(f"[Info] CRS mismatch. Reprojecting districts to match df ({df.crs})")
        districts = districts.to_crs(df.crs)

    # Sort and extract origin/destination
    df = df.sort_values(by=[trip_id_col, started_at_col])
    df["origin"] = df[geometry_col].apply(lambda line: Point(line.coords[0]))
    df["destination"] = df[geometry_col].apply(lambda line: Point(line.coords[-1]))

    # Create origin and destination GeoDataFrames
    origin = gpd.GeoDataFrame(
        df[[trip_id_col, started_at_col, mode_col, length_col, duration_col, speed_col, "origin"]].copy(),
        geometry="origin",
        crs=crs
    )

    destination = gpd.GeoDataFrame(
        df[[trip_id_col, finished_at_col, mode_col, length_col, duration_col, speed_col, "destination"]].copy(),
        geometry="destination",
        crs=crs
    )

    # Spatial joins
    try:
        origin_areas = origin.sjoin(districts, how="left", predicate="within")
        destination_areas = destination.sjoin(districts, how="left", predicate="within")
    except Exception as e:
        raise RuntimeError(f"Spatial join failed: {e}")

    # Compose final OD flow DataFrame
    od_flows = origin[[trip_id_col, "origin", started_at_col]].copy()
    od_flows["destination"] = destination_areas["destination"]
    od_flows["finished_at"] = destination_areas[finished_at_col]
    od_flows["origin_area"] = origin_areas.get(area_col)
    od_flows["dest_area"] = destination_areas.get(area_col)
    od_flows["origin_antenna"] = origin_areas.get(antenna_col)
    od_flows["dest_antenna"] = destination_areas.get(antenna_col)
    od_flows["origin_mode"] = origin_areas.get(mode_col)
    od_flows["dest_mode"] = destination_areas.get(mode_col)
    od_flows["duration_minutes"] = destination_areas.get(duration_col)
    od_flows["speed"] = destination_areas.get(speed_col)
    od_flows["origin_length"] = origin_areas.get(length_col)
    od_flows["dest_length"] = destination_areas.get(length_col)

    # Final filtering
    #od_flows = od_flows[
        #(od_flows["origin_antenna"] != od_flows["dest_antenna"]) &
        #(od_flows["dest_mode"].notna())
    #]

    return od_flows


def summarize_eventlog(log):
    n_cases = len(log)
    n_events = sum(len(trace) for trace in log)
    activities = set()
    for trace in log:
        for ev in trace:
            if 'concept:name' in ev:
                activities.add(ev['concept:name'])

    # events per case
    ev_per_case = [len(t) for t in log]

    # durations (needs time:timestamp)
    import pandas as pd
    durs = []
    for t in log:
        ts = [ev['time:timestamp'] for ev in t if 'time:timestamp' in ev]
        if ts:
            durs.append(max(ts) - min(ts))
    duration_stats = pd.Series(durs).describe() if durs else "no timestamps"

    print(f"cases: {n_cases}")
    print(f"events: {n_events}")
    print(f"activities: {len(activities)}")
    print("events per case (count/mean/std/min/25%/50%/75%/max):")
    print(pd.Series(ev_per_case).describe())
    print("\ncase duration stats:")
    print(duration_stats)
