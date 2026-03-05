#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_univariate.py
Create a NATIONAL univariate monthly overdose time series (2010-2023)
by combining two CDC WONDER exports (2010–2017 and 2018–2023).

Output format mimics your existing univariate file:
- Row Labels: Timestamp first-of-month
- Month: Jan..Dec
- Month_Code: 1..12
- Year_Code: 4-digit year
- Sum of Deaths: monthly total (national)

Saves CSV and XLSX into /mnt/data/processed_data by default.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------ #
# Update these paths if needed.
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File paths for sex-aggregated data
FILE_2018_2023_SEX = os.path.join(RAW_DATA_DIR, 'Agg_Sex_Year_Month.xlsx')
FILE_2010_2017_SEX = os.path.join(RAW_DATA_DIR, 'Agg_Sex_Year_Month_2017.xlsx')

# File paths for state-aggregated data
FILE_2018_2023_STATE = os.path.join(RAW_DATA_DIR, 'Agg_State_Year_Month.xlsx')
FILE_2010_2017_STATE = os.path.join(RAW_DATA_DIR, 'Agg_State_Year_Month_2017.xlsx')

OUT_CSV  = os.path.join(PROCESSED_DATA_DIR, "national_month_overdose_2010_2023.csv")
OUT_XLSX = os.path.join(PROCESSED_DATA_DIR, "national_month_overdose_2010_2023.xlsx")

# ------------------ UTILITIES ------------------ #
def read_excel_robust(path):
    """
    Try to read Excel using openpyxl (xlsx) first, then xlrd (xls).
    If both fail, instruct the user to install xlrd<2.0 or re-save file as .xlsx.
    """
    ext = os.path.splitext(path)[1].lower()
    # Prefer openpyxl for xlsx
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path, engine="openpyxl")
    # Try openpyxl even if misnamed
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read '{path}'. If it's a true .xls, install xlrd<2.0 "
                f"(pip install 'xlrd<2.0') or re-save as .xlsx. Original error: {repr(e)}"
            )

def parse_date_column(df):
    """
    Parse first-of-month timestamps from common CDC WONDER columns.
    """
    cols = {c.lower(): c for c in df.columns}
    # 1) Try obvious date-like columns
    for cand in ["month code", "month", "year-month", "date"]:
        if cand in cols:
            dates = pd.to_datetime(df[cols[cand]], errors="coerce")
            if dates.notna().sum() > 0:
                return dates.dt.to_period("M").dt.to_timestamp()
    # 2) Build from Year + Month_Code/Month
    year_col = cols.get("year code") or cols.get("year")
    month_col = cols.get("month code") or cols.get("month")
    if year_col and month_col:
        year_vals = pd.to_numeric(df[year_col], errors="coerce")
        month_dt = pd.to_datetime(df[month_col], errors="coerce")
        month_vals = month_dt.dt.month
        if month_vals.isna().all():
            month_vals = pd.to_numeric(df[month_col], errors="coerce")
        out = pd.to_datetime(
            year_vals.astype("Int64").astype(str).str.zfill(4) + "-" +
            month_vals.astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )
        return out.dt.to_period("M").dt.to_timestamp()
    raise ValueError("Could not parse monthly dates from available columns.")

def find_deaths_column(df):
    for cand in ["Deaths", "Death", "Number of Deaths", "Deaths Count", "Count", "Sum of Deaths"]:
        if cand in df.columns:
            return cand
    # fallback: look for any column containing "death"
    for c in df.columns:
        if "death" in c.lower():
            return c
    raise ValueError(f"No deaths-like column found in columns: {list(df.columns)}")

def clean_deaths(series):
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.replace({
        "Suppressed": np.nan,
        "Not Applicable": np.nan,
        "NaN": np.nan,
        "nan": np.nan,
        "": np.nan
    })
    return pd.to_numeric(s, errors="coerce")

def to_complete_months(df):
    """Ensure dense monthly index from global min..max."""
    df = df.copy().dropna(subset=["Month"])
    min_m = df["Month"].min()
    max_m = df["Month"].max()
    months = pd.period_range(min_m, max_m, freq="M").to_timestamp()
    return pd.DataFrame({"Month": months}).merge(df, on="Month", how="left")

def load_any_pair(file18_23, file10_17, fallback_msg):
    """
    Load a pair (2018-23 and 2010-17). Return combined df with ['Month','Deaths'].
    If a file is missing, returns None.
    """
    if not (os.path.exists(file18_23) and os.path.exists(file10_17)):
        print(fallback_msg)
        return None
    print(f"Loading {file10_17} ...")
    df_old = read_excel_robust(file10_17)
    print(f"Loading {file18_23} ...")
    df_new = read_excel_robust(file18_23)

    def norm(df):
        df = df.copy()
        deaths_col = find_deaths_column(df)
        df["Deaths"] = clean_deaths(df[deaths_col])
        df["Month"]  = parse_date_column(df)
        return df[["Month", "Deaths"]].dropna(subset=["Month"])

    df_old = norm(df_old)
    df_new = norm(df_new)
    both = pd.concat([df_old, df_new], ignore_index=True)
    both = both.sort_values("Month").drop_duplicates(subset=["Month"], keep="first").reset_index(drop=True)
    # densify and interpolate
    both = to_complete_months(both)
    both["Deaths"] = both["Deaths"].interpolate("linear", limit_direction="both").ffill().bfill()
    return both

def build_national_monthly():
    """
    Prefer STATE files (sum across states) if present; else use SEX files (sum across sexes).
    If both are present, we simply use STATE pair.
    """
    have_state = os.path.exists(FILE_2018_2023_STATE) and os.path.exists(FILE_2010_2017_STATE)
    have_sex   = os.path.exists(FILE_2018_2023_SEX) and os.path.exists(FILE_2010_2017_SEX)

    if not (have_state or have_sex):
        raise FileNotFoundError(
            "No input files found. Expected either the STATE pair or the SEX pair in /mnt/data."
        )

    if have_state:
        print("Using STATE-aggregated inputs (preferred).")
        # For state files, we may need to SUM across states per month.
        # We'll read both, normalize, then groupby Month to sum.
        def load(file18_23, file10_17):
            print(f"Loading {file10_17} ...")
            df_old = read_excel_robust(file10_17)
            print(f"Loading {file18_23} ...")
            df_new = read_excel_robust(file18_23)
            return df_old, df_new

        df_old, df_new = load(FILE_2018_2023_STATE, FILE_2010_2017_STATE)

        def norm_group(df):
            df = df.copy()
            deaths_col = find_deaths_column(df)
            df["Deaths"] = clean_deaths(df[deaths_col])
            df["Month"]  = parse_date_column(df)
            return df[["Month", "Deaths"]].groupby("Month", as_index=False)["Deaths"].sum()

        df_old = norm_group(df_old)
        df_new = norm_group(df_new)
        both = pd.concat([df_old, df_new], ignore_index=True)
    else:
        print("Using SEX-aggregated inputs.")
        # For sex files, sum across sexes per month.
        both = load_any_pair(FILE_2018_2023_SEX, FILE_2010_2017_SEX,
                             "SEX files missing; cannot build from sexes.")
        if both is None:
            raise FileNotFoundError("SEX pair missing; cannot proceed.")

    # Deduplicate, densify, interpolate
    both = both.sort_values("Month").drop_duplicates(subset=["Month"], keep="first").reset_index(drop=True)
    both = to_complete_months(both)
    both["Deaths"] = both["Deaths"].interpolate("linear", limit_direction="both").ffill().bfill()

    # Final tidy frame in the requested style
    out = pd.DataFrame()
    out["Row Labels"] = both["Month"]
    out["Month"]      = out["Row Labels"].dt.strftime("%b")  # Jan..Dec
    out["Month_Code"] = out["Row Labels"].dt.month
    out["Year_Code"]  = out["Row Labels"].dt.year
    out["Sum of Deaths"] = both["Deaths"].round(0).astype(int)

    # Sort by date just in case
    out = out.sort_values("Row Labels").reset_index(drop=True)

    # Save
    out.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="national_monthly")

    print(f"\nSaved CSV:  {OUT_CSV}")
    print(f"Saved XLSX: {OUT_XLSX}")
    print("\nPreview:")
    print(out.head(12))

def main():
    out = build_national_monthly()

if __name__ == "__main__":
    main()
