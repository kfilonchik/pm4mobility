import pandas as pd

def create_ocel_from_dataframe(
    df,
    trip_id_col="trip_id",
    start_time_col="started_at",
    end_time_col="finished_at",
    origin_area_col="origin_area",
    dest_area_col="dest_area",
    origin_mode_col="origin_mode",
    dest_mode_col="dest_mode",
    origin_event_col="origin_antenna",
    dest_event_col="dest_antenna"
):
    """
    Create an OCEL structure where:
    - Object types = all unique area names + all unique mode names
    - Object IDs = area_i for areas; mode_tripid for trips
    - Events relate to (area object, qualifier = area name) and (trip object, qualifier = mode)
    """
    ocel = {
        "objectTypes": [],
        "objects": [],
        "eventTypes": [],
        "events": []
    }

    # 1. Object types: unique area names + mode names
    #area_names = pd.Series(sorted(set(df[origin_area_col]) | set(df[dest_area_col]))).dropna().unique()
    mode_names = pd.Series(sorted(set(df[origin_mode_col]) | set(df[dest_mode_col]))).dropna().unique()

    ocel["objectTypes"] = (
        #[{"name": area_name, "attributes": []} for area_name in area_names] +
        [{"name": mode, "attributes": []} for mode in mode_names]
    )

    # 2. Object IDs

    # Area objects: area_1, area_2, ... typed by area name
    #area_id_map = {area: f"area_{i+1}" for i, area in enumerate(area_names)}
    #for area_name, area_id in area_id_map.items():
       # ocel["objects"].append({
          # "id": area_id,
          #  "type": area_name,
          #  "attributes": []
       # })

    # Mode objects: one per trip, ID = mode_tripid, type = mode
    trip_ids = set()
    for _, row in df.iterrows():
        origin_trip_id = f"{row[origin_mode_col]}_{row[trip_id_col]}"
        dest_trip_id = f"{row[dest_mode_col]}_{row[trip_id_col]}"
        trip_ids.add((origin_trip_id, row[origin_mode_col]))
        trip_ids.add((dest_trip_id, row[dest_mode_col]))

    for trip_id, mode in trip_ids:
        ocel["objects"].append({
            "id": trip_id,
            "type": mode,
            "attributes": []
        })

    # 3. Event types
    all_event_types = pd.Series(sorted(set(df[origin_event_col]) | set(df[dest_event_col]))).dropna().unique()
    ocel["eventTypes"] = [{"name": ev, "attributes": []} for ev in all_event_types]

    # 4. Events
    event_counter = 1
    for _, row in df.iterrows():
        # Trip object IDs
        origin_trip_id = f"{row[origin_mode_col]}_{row[trip_id_col]}"
        dest_trip_id = f"{row[dest_mode_col]}_{row[trip_id_col]}"
        # Area object IDs
        #origin_area_id = area_id_map.get(row[origin_area_col])
        #dest_area_id = area_id_map.get(row[dest_area_col])

        # Origin event
        ocel["events"].append({
            "id": f"e{event_counter}",
            "type": row[origin_area_col],
            "time": pd.to_datetime(row[start_time_col]).isoformat(),
            "relationships": [
                {"objectId": origin_trip_id, "qualifier": row[origin_mode_col]},
                #{"objectId": origin_area_id, "qualifier": row[origin_area_col]}
            ]
        })
        event_counter += 1

        # Destination event
        ocel["events"].append({
            "id": f"e{event_counter}",
            "type": row[dest_area_col],
            "time": pd.to_datetime(row[end_time_col]).isoformat(),
            "relationships": [
                {"objectId": dest_trip_id, "qualifier": row[dest_mode_col]},
                #{"objectId": dest_area_id, "qualifier": row[dest_area_col]}
            ]
        })
        event_counter += 1

    return ocel