import pandas as pd

def create_trad_event_log(
    df: pd.DataFrame,
    case_id: str = "trip_id",
    start_time: str = "started_at",
    end_time: str = "finished_at",
    event_origin: str = "origin_area",
    event_dest: str = "dest_area",
    mode_origin: str = "origin_mode",
    mode_dest: str = "dest_mode",
    length_origin: str = "origin_length",
    length_dest: str = "dest_length",
    area_origin: str = "origin_area",
    area_dest: str = "dest_area",
    antenna_origin: str = "origin_antenna",
    antenna_dest: str = "dest_antenna"
) -> pd.DataFrame:
    """
    Create a traditional event log from OD flows by flattening origin and destination into events.

    Parameters:
    - df: DataFrame containing OD flow records
    - All other parameters are column names for origin/destination features

    Returns:
    - DataFrame formatted for traditional process mining:
      columns: caseid, user_id, timestamp, event, area, mode, length
    """

    required_cols = [
        case_id,
        start_time, end_time,
        event_origin, event_dest,
        mode_origin, mode_dest,
        length_origin, length_dest,
        area_origin, area_dest,
        antenna_origin, antenna_dest
    ]

    # Check if all required columns exist
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input DataFrame: {missing}")

    try:
        # Select and rename origin
        origin = df[[case_id, start_time, event_origin, antenna_origin, length_origin, mode_origin]].copy()
        origin.columns = ["caseid", "timestamp", "event", "antenna", "length", "mode"]

        # Select and rename destination
        destination = df[[case_id, end_time, event_dest, antenna_dest, length_dest, mode_dest]].copy()
        destination.columns = ["caseid", "timestamp", "event", "antenna", "length", "mode"]

        # Concatenate and sort
        event_log = pd.concat([origin, destination], ignore_index=True)
        event_log.sort_values(by=["caseid", "timestamp"], inplace=True)

        return event_log

    except Exception as e:
        raise RuntimeError(f"Failed to generate traditional event log: {e}")
