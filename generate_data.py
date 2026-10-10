"""Generate simulated sensor data for industrial pumps.

Each machine runs normally, then slowly degrades until it fails.
During degradation, vibration and temperature rise and pressure drops.
Output: data/sensor_data.csv

NOTE: this data is SIMULATED. We say so clearly in the README and the demo.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed -> same data every run

N_MACHINES = 40
rows = []

for machine_id in range(1, N_MACHINES + 1):
    life = int(rng.integers(180, 320))        # hours until failure
    degrade_start = life - int(rng.integers(50, 90))  # when wear begins

    for t in range(1, life + 1):
        # wear goes 0 -> 1 during the degradation period
        wear = max(0.0, (t - degrade_start) / (life - degrade_start))

        temperature = 65 + rng.normal(0, 1.2) + 14 * wear**1.5
        vibration = 2.0 + rng.normal(0, 0.15) + 3.2 * wear**2
        pressure = 5.0 + rng.normal(0, 0.10) - 1.1 * wear
        rpm = 1500 + rng.normal(0, 12) - 60 * wear
        power_kw = 18 + rng.normal(0, 0.5) + 4.0 * wear

        rows.append({
            "machine_id": machine_id,
            "hour": t,
            "temperature_c": round(temperature, 2),
            "vibration_mm_s": round(vibration, 3),
            "pressure_bar": round(pressure, 3),
            "rpm": round(rpm, 1),
            "power_kw": round(power_kw, 2),
            "hours_to_failure": life - t,
        })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sensor_data.csv", index=False)
print(f"Saved {len(df)} rows for {N_MACHINES} machines to data/sensor_data.csv")
print(df.head(3).to_string(index=False))
