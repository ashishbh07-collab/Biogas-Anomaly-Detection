"""
DENG611 Assessment 2: Biogas process anomaly detection with Isolation Forest.

Input:
    Farm-scale Biodigester 2024_Performance.xlsx
Output:
    cleaned_biogas_data.csv
    anomaly_detection_test_results.csv
    figures/figure_1_timeseries_anomalies.png
    figures/figure_2_anomaly_scores.png
    figures/figure_3_temperature_vs_biogas.png
    figures/figure_4_biogas_by_digester.png

Install:
    pip install pandas openpyxl scikit-learn matplotlib

Run this script in the same folder as the Excel workbook:
    python biogas_anomaly_detection.py

Important:
- This is unsupervised anomaly detection, not a verified fault classifier.
- The data cleaning rules below are transparent screening rules for implausible
  entries and should be checked against the original field/lab records.
- Because confirmed fault labels are not available, precision, recall, F1 and
  accuracy are not calculated.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

INPUT_FILE = Path("Farm-scale Biodigester 2024_Performance.xlsx")
OUTPUT_DIR = Path("biogas_anomaly_results")
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Excel sheet and column names. R1 is labelled "RI-FLEX" (letter I) in the workbook.
SHEETS = [
    ("RI-FLEX", "RI-FLEX", "DigesterTemp (C)09:25am"),
    ("R2-FLEX", "R2-FLEX", "DigesterTemp (C)time 09:25am"),
]

def numeric_feed(series):
    """Extract a numeric feed quantity from entries such as '20kg'."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )

frames = []
for sheet, label, digester_temp_col in SHEETS:
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet)
    date_col = "Date (m/d/y)"
    if date_col not in df.columns:
        raise ValueError(f"Expected date column {date_col!r} in sheet {sheet!r}")

    frame = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "digester": label,
        "air_temp_c": pd.to_numeric(df["Air Temp (C)time 09:15am"], errors="coerce"),
        "digester_temp_c": pd.to_numeric(df[digester_temp_col], errors="coerce"),
        "biogas_stp_ml": pd.to_numeric(df["Daily Biogas at STP (mL)"], errors="coerce"),
        "water_added_kg": pd.to_numeric(df["Water added (kg)"], errors="coerce"),
        "manure_fed_kg": numeric_feed(df["Manure fed (kg)"]),
    })
    frames.append(frame)

raw = pd.concat(frames, ignore_index=True)
raw = raw.dropna(subset=["date"]).copy()

# Screening / cleaning:
# 1) Air temperature 10-45 C and digester temperature 10-40 C are retained.
#    Values outside these ranges are treated as missing rather than as real events.
# 2) Negative gas production and values >10,000 mL/day are treated as invalid
#    meter/reset/calculation entries and set to missing. The threshold is a
#    transparent data-quality screen, not a universal engineering limit.
data = raw.copy()
data.loc[~data["air_temp_c"].between(10, 45), "air_temp_c"] = np.nan
data.loc[~data["digester_temp_c"].between(10, 40), "digester_temp_c"] = np.nan
data.loc[~data["biogas_stp_ml"].between(0, 10000), "biogas_stp_ml"] = np.nan

# Keep records with valid gas measurement for the main analysis. Missing
# temperature/feed values are median-imputed within the training pipeline.
data = data.dropna(subset=["biogas_stp_ml"])
data = data.sort_values(["date", "digester"]).reset_index(drop=True)

features = [
    "air_temp_c",
    "digester_temp_c",
    "biogas_stp_ml",
    "water_added_kg",
    "manure_fed_kg",
]

# Chronological 70/30 split. Do not shuffle time-series data.
split_idx = int(len(data) * 0.70)
train = data.iloc[:split_idx].copy()
test = data.iloc[split_idx:].copy()
if len(train) < 20 or len(test) < 10:
    raise ValueError("Not enough usable records after cleaning for a stable split.")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )),
])
pipeline.fit(train[features])

# IsolationForest: +1 = inlier, -1 = anomaly.
test["prediction"] = pipeline.predict(test[features])
test["anomaly_flag"] = test["prediction"].eq(-1)
# Higher score means more anomalous; the threshold is 0 for this transformed
# decision score because sklearn's decision_function uses 0 as its boundary.
test["anomaly_score"] = -pipeline.decision_function(test[features])
train["prediction"] = pipeline.predict(train[features])
train["anomaly_flag"] = train["prediction"].eq(-1)
train["anomaly_score"] = -pipeline.decision_function(train[features])

# Save reusable analysis tables.
data.to_csv(OUTPUT_DIR / "cleaned_biogas_data.csv", index=False)
test.to_csv(OUTPUT_DIR / "anomaly_detection_test_results.csv", index=False)

# Summary printed to terminal.
print("=== Data summary ===")
print(f"Rows after combining selected sheets (before screening): {len(raw)}")
print(f"Rows with valid gas measurement after screening: {len(data)}")
print(f"Rows removed for invalid/missing gas measurement: {len(raw) - len(data)}")
print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"Training rows: {len(train)}; test rows: {len(test)}")
print(f"Training anomalies: {int(train['anomaly_flag'].sum())} "
      f"({train['anomaly_flag'].mean():.1%})")
print(f"Test anomalies: {int(test['anomaly_flag'].sum())} "
      f"({test['anomaly_flag'].mean():.1%})")
print("\\nGas summary by digester (mL/day at STP):")
print(data.groupby("digester")["biogas_stp_ml"].agg(
    ["count", "mean", "median", "min", "max"]
).round(2))
print("\\nNote: no verified fault labels were available; classification metrics are not valid.")

# Figure 1: chronological gas output with model-flagged test anomalies.
fig, ax = plt.subplots(figsize=(12, 6))
for name, group in data.groupby("digester"):
    ax.plot(group["date"], group["biogas_stp_ml"], linewidth=1, alpha=0.75, label=f"{name} daily biogas")
test_anom = test[test["anomaly_flag"]]
ax.scatter(test_anom["date"], test_anom["biogas_stp_ml"], marker="x", s=55,
           label="Test observations flagged as anomalies", zorder=4)
ax.axvline(test["date"].min(), linestyle="--", linewidth=1.2, label="Test period begins")
ax.set_title("Figure 1. Daily biogas production and detected test anomalies")
ax.set_xlabel("Date")
ax.set_ylabel("Daily biogas at STP (mL/day)")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_1_timeseries_anomalies.png", dpi=180)
plt.close(fig)

# Figure 2: test anomaly scores and decision boundary.
fig, ax = plt.subplots(figsize=(10, 5.5))
colours = np.where(test["anomaly_flag"], "tab:red", "tab:blue")
ax.scatter(np.arange(len(test)), test["anomaly_score"], c=colours, s=24, alpha=0.8)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="Isolation Forest decision boundary")
ax.set_title("Figure 2. Isolation Forest anomaly scores for chronological test set")
ax.set_xlabel("Test observation index (chronological order)")
ax.set_ylabel("Anomaly score = −decision_function (higher = more unusual)")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_2_anomaly_scores.png", dpi=180)
plt.close(fig)

# Figure 3: relation between air temperature and gas production.
fig, ax = plt.subplots(figsize=(8.5, 6))
normal = test[~test["anomaly_flag"]]
anomalies = test[test["anomaly_flag"]]
ax.scatter(normal["air_temp_c"], normal["biogas_stp_ml"], alpha=0.65, s=30, label="Not flagged")
ax.scatter(anomalies["air_temp_c"], anomalies["biogas_stp_ml"], marker="x", s=65,
           label="Flagged anomaly")
ax.set_title("Figure 3. Air temperature versus daily biogas production (test data)")
ax.set_xlabel("Air temperature (°C)")
ax.set_ylabel("Daily biogas at STP (mL/day)")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_3_temperature_vs_biogas.png", dpi=180)
plt.close(fig)

# Figure 4: compare cleaned gas production distributions by flexible digester.
fig, ax = plt.subplots(figsize=(8.5, 5.5))
groups = [data.loc[data["digester"] == name, "biogas_stp_ml"].dropna().values
          for name in ["RI-FLEX", "R2-FLEX"]]
ax.boxplot(groups, tick_labels=["RI-FLEX", "R2-FLEX"], showmeans=True)
ax.set_title("Figure 4. Distribution of cleaned daily biogas production by digester")
ax.set_xlabel("Digester")
ax.set_ylabel("Daily biogas at STP (mL/day)")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "figure_4_biogas_by_digester.png", dpi=180)
plt.close(fig)

print(f"\\nResults and figures saved in: {OUTPUT_DIR.resolve()}")
