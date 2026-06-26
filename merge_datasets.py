"""
merge_datasets.py
=================
Merges per-year feature CSVs into a single master dataset.

Input:  data/features_2022.csv, data/features_2023.csv, data/features_2024.csv
Output: data/features_master.csv

Usage
-----
    python merge_datasets.py
"""

import pandas as pd

INPUT_FILES = [
    "data/features_2022.csv",
    "data/features_2023.csv",
    "data/features_2024.csv",
]
OUTPUT_FILE = "data/features_master.csv"

# ── Load ──────────────────────────────────────────────────────────────
dfs = [pd.read_csv(f) for f in INPUT_FILES]
for path, df in zip(INPUT_FILES, dfs):
    print(f"Loaded {path}  ({len(df):,} rows)")

# ── Concatenate vertically ────────────────────────────────────────────
master = pd.concat(dfs, ignore_index=True)

# ── Sort chronologically per coordinate ───────────────────────────────
master.sort_values(["lat", "lon", "day_of_year"], inplace=True)
master.reset_index(drop=True, inplace=True)

# ── Save ──────────────────────────────────────────────────────────────
master.to_csv(OUTPUT_FILE, index=False)
print(f"\n[OK] Saved -> {OUTPUT_FILE}  ({len(master):,} rows)")
