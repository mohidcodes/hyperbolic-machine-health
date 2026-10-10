"""Train the early-warning model and measure how well it works.

Idea: label every hour as "at risk" if the machine fails within the next
30 hours. The model learns to spot the sensor pattern that comes before failure.

Output: models/risk_model.joblib and printed results.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

SENSORS = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm", "power_kw"]
RISK_WINDOW = 30  # "at risk" = fails within this many hours


def add_features(df):
    """Raw sensors + the average of the last 10 hours (smooths out noise)."""
    df = df.sort_values(["machine_id", "hour"]).copy()
    for col in SENSORS:
        df[col + "_avg10"] = (
            df.groupby("machine_id")[col]
            .transform(lambda s: s.rolling(10, min_periods=1).mean())
        )
    return df


def feature_columns():
    return SENSORS + [c + "_avg10" for c in SENSORS]


if __name__ == "__main__":
    df = add_features(pd.read_csv("data/sensor_data.csv"))
    df["at_risk"] = (df["hours_to_failure"] <= RISK_WINDOW).astype(int)

    # Split by MACHINE, so the model is tested on machines it has never seen.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(splitter.split(df, groups=df["machine_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx].copy()

    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=0, class_weight="balanced"
    )
    model.fit(train[feature_columns()], train["at_risk"])

    test["risk"] = model.predict_proba(test[feature_columns()])[:, 1]
    pred = (test["risk"] >= 0.5).astype(int)

    print(f"Trained on {train['machine_id'].nunique()} machines, "
          f"tested on {test['machine_id'].nunique()} unseen machines")
    print(f"Precision: {precision_score(test['at_risk'], pred):.2f}  "
          f"(when it warns, how often it is right)")
    print(f"Recall:    {recall_score(test['at_risk'], pred):.2f}  "
          f"(how many at-risk hours it catches)")

    # The number judges care about: how much warning do operators get?
    leads, false_alarms = [], 0
    for _, m in test.groupby("machine_id"):
        m = m.sort_values("hour")
        # alarm = risk >= 0.5 for 3 hours in a row
        alarm = (m["risk"] >= 0.5).rolling(3).sum() == 3
        if alarm.any():
            first = m[alarm].iloc[0]
            leads.append(int(first["hours_to_failure"]))
        # healthy period: more than 90 hours before failure
        false_alarms += int(((m["risk"] >= 0.5) & (m["hours_to_failure"] > 90)).sum())

    n_machines = test["machine_id"].nunique()
    print(f"Machines warned before failure: {len(leads)} of {n_machines}")
    if leads:
        print(f"Average warning time: {np.mean(leads):.0f} hours before failure "
              f"(shortest: {min(leads)} hours)")
    print(f"False alarm hours during healthy operation: {false_alarms}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/risk_model.joblib")
    print("Saved models/risk_model.joblib")
