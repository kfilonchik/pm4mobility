import pandas as pd
import numpy as np

def estimate_transport_mode(distance_m, duration_min, speed_mps):
    """
    Estimate transport mode based on distance, duration, and speed (m/s).
    
    Parameters:
    - distance_m: float or pd.Series, distance in meters
    - duration_min: float or pd.Series, duration in minutes
    - speed_mps: float or pd.Series, speed in meters per second

    Returns:
    - A transport mode label (str or pd.Series)
    """
    # Convert single values to Series for unified logic
    is_scalar = np.isscalar(speed_mps)
    
    if is_scalar:
        data = pd.DataFrame({
            "distance": [distance_m],
            "duration": [duration_min],
            "speed": [speed_mps]
        })
    else:
        data = pd.DataFrame({
            "distance": distance_m,
            "duration": duration_min,
            "speed": speed_mps
        })

    def classify(row):
        d, t, s = row["distance"], row["duration"], row["speed"]
        if pd.isna(d) or pd.isna(t) or pd.isna(s) or t == 0:
            return "unknown"
        if s < 2 and t < 80:
            return "foot"
        elif 2 <= s < 5 and t < 120:
            return "bicycle"
        elif 5 <= s < 8 and 10 <= t <= 80:
            return "bus"
        elif 8 <= s < 12 and 5 <= t <= 70:
            return "car"
        elif 12 <= s < 25 and 5 <= t <= 60:
            return "train_or_metro"
        elif s >= 25 and t <= 120:
            return "long_distance_train"
        else:
            return "unknown"

    result = data.apply(classify, axis=1)
    return result.iloc[0] if is_scalar else result
