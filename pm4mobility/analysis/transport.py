import pandas as pd
import numpy as np
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

def estimate_transport_mode_simple(distance_m, duration_min, speed_mps):
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

# ---- conservative seed labels (high precision, low recall) ----
CLASSES = ["foot","bicycle","bus","car","tram_metro","train"]
# ---- build features ----
def build_features(df):
    X = df.copy()
    X["started_at"]  = pd.to_datetime(X["started_at"], utc=True, errors="coerce")
    X["finished_at"] = pd.to_datetime(X["finished_at"], utc=True, errors="coerce")
    # If you only have haversine length, apply a small detour factor to not under-estimate speed
    if "duration_min" not in X or X["duration_min"].isna().any():
        dur_sec = (X["finished_at"] - X["started_at"]).dt.total_seconds()
        X["duration_min"] = dur_sec.where(dur_sec > 0, np.nan) / 60.0
    if "speed_mps" not in X or X["speed_mps"].isna().any():
        detour = 1.4 # ~25% longer than straight line; set to 1.0 if you have path length
        X["speed_mps"] = (X["length"] * detour) / (X["duration_min"] * 60)
    X["kmh"] = X["speed_mps"] * 3.6
    X["start_hour"] = X["started_at"].dt.hour
    X["weekday"] = X["started_at"].dt.dayofweek
    X["is_night"] = ((X["start_hour"] < 6) | (X["start_hour"] >= 22)).astype(int)

    # Ensure optional columns exist
    for c in ["straightness","stop_rate","rail_proximity"]:
        if c not in X.columns: X[c] = np.nan
    return X



def make_seeds(X):
    y_seed = np.full(len(X), fill_value=-1, dtype=object)  # -1 = unlabeled (required by SelfTraining)
    kmh = X["kmh"].values
    dkm = (X["length"].values / 1000.0)
    sr  = X["stop_rate"].fillna(0).values
    rail= X["rail_proximity"].fillna(0).values
    straight = X["straightness"].fillna(np.nan).values

    # FOOT: very slow & short
    y_seed[(kmh <= 5.5) & (dkm <= 3.0)] = "foot"

    # BICYCLE: 10–28 km/h, modest distances, not too straight like rail
    y_seed[(kmh >= 10) & (kmh <= 28) & (dkm <= 20) & (rail < 0.5)] = "bicycle"

    # BUS: 8–22 km/h AND many stops/km OR long urban distances
    y_seed[((kmh >= 8) & (kmh <= 22)) & ((sr >= 0.8) | (dkm >= 3))] = "bus"

    # CAR: > 25 km/h up to ~120, not along rail, low stop density
    y_seed[(kmh >= 25) & (kmh <= 120) & (rail < 0.5) & (sr <= 0.7)] = "car"

    # TRAM/METRO: 20–40 km/h, near rail, medium distances
    y_seed[(kmh >= 20) & (kmh <= 40) & (rail >= 0.5)] = "tram_metro"

    # TRAIN: ≥ 45 km/h OR (near rail & very straight & ≥5 km)
    y_seed[(kmh >= 45)] = "train"
    y_seed[(rail >= 0.5) & (dkm >= 5) & (~np.isnan(straight)) & (straight >= 0.9)] = "train"

    # return as categorical with -1 for unlabeled
    return pd.Series(y_seed, index=X.index)

# ---- choose columns for the model ----
NUM_COLS = ["speed_mps","kmh","duration_min","length",
            "straightness","stop_rate","rail_proximity","start_hour"]
CAT_COLS = ["weekday","is_night"]

def train_self_training(df, groups=None, seed_threshold=0.92):
    X = build_features(df)
    y_seed = make_seeds(X)

    
    NUM_COLS = ["speed_mps","kmh","duration_min","length",
                "straightness","stop_rate","rail_proximity","start_hour"]
    CAT_COLS = ["weekday","is_night"]

    feature_cols = NUM_COLS + CAT_COLS
    Xf = X[feature_cols].copy()                   # keep as DataFrame to get positions
    num_idx = [Xf.columns.get_loc(c) for c in NUM_COLS]
    cat_idx = [Xf.columns.get_loc(c) for c in CAT_COLS]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_idx),
        ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.int8), cat_idx),
    ])

    #base = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=400)

    base = HistGradientBoostingClassifier(
        learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=0.1, early_stopping=True
    )

    pipe = Pipeline([("pre", pre), ("clf", base)])

    #st = SelfTrainingClassifier(base_estimator=base, threshold=0.92, verbose=False)  # ↑ threshold

    st = SelfTrainingClassifier(base_estimator=pipe, threshold=0.92, verbose=True)

    # IMPORTANT: pass a **NumPy array** now that we use indices
    st.fit(Xf.values, y_seed.values)

    proba = st.predict_proba(Xf.values)
    labels  = st.predict(Xf.values)
    return labels, proba, y_seed


def smooth_labels(df, labels_col="pred", user_col="user_id", time_col="started_at", k=3):
    # force odd window
    k = int(k);  k = k if k % 2 == 1 else k + 1

    s = df[[user_col, time_col, labels_col]].copy()
    s[time_col] = pd.to_datetime(s[time_col], utc=True)
    s = s.sort_values([user_col, time_col])

    # encode labels -> category codes
    cats = pd.Categorical(s[labels_col])
    codes = pd.Series(cats.codes, index=s.index)  # 0..C-1

    # one-hot over codes
    oh = pd.get_dummies(codes).astype("int16")

    # rolling sums per user (centered window)
    roll = oh.groupby(s[user_col]).apply(
        lambda g: g.rolling(k, center=True, min_periods=1).sum()
    )
    # drop the added group level in index
    if isinstance(roll.index, pd.MultiIndex):
        roll = roll.droplevel(0)

    # pick class with max votes; on ties keep the current label
    max_per_row = roll.max(axis=1)
    is_tie = roll.eq(max_per_row, axis=0).sum(axis=1) > 1
    argmax_code = roll.idxmax(axis=1)

    # current (center) code = original label at that row
    current_code = codes
    sm_code = np.where(is_tie.values, current_code.values, argmax_code.values)

    # map codes -> labels
    smoothed = pd.Series(pd.Categorical.from_codes(sm_code, cats.categories), index=s.index).astype(str)

    # align back to original df order and fill any NAs with original labels
    smoothed = smoothed.reindex(df.sort_values([user_col, time_col]).index)
    smoothed = smoothed.reindex(df.index).fillna(df[labels_col])

    return smoothed


