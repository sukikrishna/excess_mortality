import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------ #
RAW_DATA_DIR = 'raw_data'
PROCESSED_DATA_DIR = 'processed_data'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File paths for sex-aggregated data
SEX_FILE_2018_2023 = os.path.join(RAW_DATA_DIR, 'Agg_Sex_Year_Month.xlsx')
SEX_FILE_2010_2017 = os.path.join(RAW_DATA_DIR, 'Agg_Sex_Year_Month_2017.xlsx')

# File paths for state-aggregated data
STATE_FILE_2018_2023 = os.path.join(RAW_DATA_DIR, 'Agg_State_Year_Month.xlsx')
STATE_FILE_2010_2017 = os.path.join(RAW_DATA_DIR, 'Agg_State_Year_Month_2017.xlsx')

# File paths for age-aggregated data (1 covariate)
AGE_FILE_2018_2023 = os.path.join(RAW_DATA_DIR, 'Agg_Age_Year_Month.xls.xlsx')
AGE_FILE_2010_2017 = os.path.join(RAW_DATA_DIR, 'Agg_Age_Year_Month_2017.xlsx')

# NEW: File paths for age×sex-aggregated data (2 covariates)
AGE_SEX_FILE_2010_2017 = os.path.join(RAW_DATA_DIR, 'Agg_Age_Sex_Year_Month_2017.xls.xlsx')
AGE_SEX_FILE_2018_2023 = os.path.join(RAW_DATA_DIR, 'Agg_Age_Sex_Year_Month.xls.xlsx')

# ------------------ HELPER FUNCTIONS ------------------ #

def _normalize_colname(s: str) -> str:
    return str(s).strip().lower().replace('-', ' ').replace('_', ' ')

def _find_group_column(df: pd.DataFrame, preferred_name: str, candidates: list) -> str:
    """
    Return the actual column name in df that matches one of the candidates.
    Priority: exact string equality (case-insensitive), then 'contains' matches.
    """
    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}
    # Exact matches first
    for cand in [preferred_name] + candidates:
        key = _normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    # Fuzzy/contains match fallback for age cases
    for norm, orig in norm_map.items():
        if 'age' in norm and 'year' not in norm:  # avoid catching Year
            if 'group' in norm:
                return orig
    for norm, orig in norm_map.items():
        if 'age' in norm:
            return orig
    # nothing found
    raise KeyError(
        f"Could not find a column for '{preferred_name}'. "
        f"Available columns: {cols}"
    )

def _find_deaths_column(df: pd.DataFrame) -> str:
    """
    Try common Deaths column names from WONDER exports.
    """
    candidates = [
        'Deaths', 'Death', 'Number of Deaths', 'Deaths Count',
        'Deaths Code', 'Deaths (Count)'
    ]
    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}
    for cand in candidates:
        if _normalize_colname(cand) in norm_map:
            return norm_map[_normalize_colname(cand)]
    # fallback: anything containing 'death'
    for norm, orig in norm_map.items():
        if 'death' in norm:
            return orig
    raise KeyError(f"Could not find a Deaths column. Available columns: {cols}")

def parse_date_column(df):
    """Parse dates from various possible column formats"""
    # Try Month Code column first (most reliable)
    if 'Month Code' in df.columns:
        dates = pd.to_datetime(df['Month Code'], errors='coerce')
        if dates.notna().sum() > 0:
            return dates
    
    # Try Month column
    if 'Month' in df.columns:
        dates = pd.to_datetime(df['Month'], errors='coerce')
        if dates.notna().sum() > 0:
            return dates
    
    # Build from Year and Month if available
    if 'Year Code' in df.columns and 'Month Code' in df.columns:
        month_vals = pd.to_datetime(df['Month Code'], errors='coerce').dt.month
        if month_vals.notna().sum() == 0:
            month_vals = pd.to_numeric(df['Month Code'], errors='coerce')
        year_vals = pd.to_numeric(df['Year Code'], errors='coerce')
        dates = pd.to_datetime(
            year_vals.astype(str) + '-' + 
            month_vals.astype(int).astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )
        return dates
    
    raise ValueError("Could not parse dates from any available columns")

def clean_deaths_column(series):
    """Clean deaths column, handling suppressed values and converting to numeric"""
    series = series.astype(str).str.replace(',', '', regex=False)
    series = series.replace({
        'Suppressed': np.nan,
        'Not Applicable': np.nan,
        'NaN': np.nan,
        'nan': np.nan,
        '': np.nan
    })
    return pd.to_numeric(series, errors='coerce')

def interpolate_missing_values(df, groupby_cols, value_col='Deaths'):
    """
    Interpolate missing values within each group using linear interpolation.
    """
    df = df.copy()
    df = df.sort_values(['Month'] + groupby_cols).reset_index(drop=True)

    def interpolate_group(group):
        group = group.copy()
        group[value_col] = group[value_col].interpolate(method='linear', limit_direction='both')
        group[value_col] = group[value_col].fillna(method='ffill').fillna(method='bfill')
        group[value_col] = group[value_col].fillna(0)
        return group
    
    if groupby_cols:
        df = df.groupby(groupby_cols, group_keys=False).apply(interpolate_group)
    else:
        df = interpolate_group(df)
    
    return df

# ------------------ 1-COVARIATE LOADER (kept) ------------------ #

def load_and_combine_files(file1, file2, groupby_col, groupby_aliases=None):
    """
    Load two Excel files, combine them, and return cleaned dataframe.
    """
    print(f"Loading {file1}...")
    df1 = pd.read_excel(file1)
    print("Columns in file1:", list(df1.columns))
    print(f"Loading {file2}...")
    df2 = pd.read_excel(file2)
    print("Columns in file2:", list(df2.columns))

    # Age aliases (for the Age-only path)
    default_age_aliases = [
        'Age', 'Age Group', 'Age Group Code',
        'Age Groups', 'Age Groups Code',
        'Five-Year Age Groups', 'Five-Year Age Groups Code',
        'Ten-Year Age Groups', 'Ten-Year Age Groups Code'
    ]
    aliases = (groupby_aliases or []) + (default_age_aliases if groupby_col.lower() == 'age' else [])

    group_col1 = _find_group_column(df1, groupby_col, aliases)
    group_col2 = _find_group_column(df2, groupby_col, aliases)
    if group_col1 != groupby_col:
        df1 = df1.rename(columns={group_col1: groupby_col})
    if group_col2 != groupby_col:
        df2 = df2.rename(columns={group_col2: groupby_col})

    # Parse dates
    df1['Month'] = parse_date_column(df1)
    df2['Month'] = parse_date_column(df2)

    # Deaths column (robust)
    death_col_1 = _find_deaths_column(df1)
    death_col_2 = _find_deaths_column(df2)
    df1['Deaths'] = clean_deaths_column(df1[death_col_1])
    df2['Deaths'] = clean_deaths_column(df2[death_col_2])

    # Select relevant columns
    cols_to_keep = ['Month', groupby_col, 'Deaths']
    df1 = df1[cols_to_keep].dropna(subset=['Month', groupby_col])
    df2 = df2[cols_to_keep].dropna(subset=['Month', groupby_col])

    # Combine & tidy
    combined = pd.concat([df2, df1], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Month', groupby_col], keep='first')
    combined = combined.sort_values(['Month', groupby_col]).reset_index(drop=True)

    print(f"Combined data shape: {combined.shape}")
    print(f"Date range: {combined['Month'].min()} to {combined['Month'].max()}")
    print(f"Unique {groupby_col}: {combined[groupby_col].nunique()}")
    print(f"Missing values before interpolation: {combined['Deaths'].isna().sum()}")

    combined = interpolate_missing_values(combined, groupby_cols=[groupby_col])

    print(f"Missing values after interpolation: {combined['Deaths'].isna().sum()}")
    return combined

def create_wide_format(df, groupby_col):
    """Convert long format to wide format for multivariate time series."""
    wide = df.pivot(index='Month', columns=groupby_col, values='Deaths')
    wide = wide.sort_index()
    wide = wide.interpolate(method='linear', limit_direction='both')
    wide = wide.fillna(method='ffill').fillna(method='bfill').fillna(0)
    print(f"\nWide format shape: {wide.shape}")
    print(f"Columns: {list(wide.columns)[:10]}{'...' if len(wide.columns) > 10 else ''}")
    print(f"Date range: {wide.index.min()} to {wide.index.max()}")
    return wide

def create_train_val_test_split(df, train_end='2019-01-01', val_end='2020-01-01'):
    """Create train/validation/test splits for multivariate data"""
    train = df[df.index < train_end]
    validation = df[(df.index >= train_end) & (df.index < val_end)]
    test = df[df.index >= val_end]
    print(f"\nTrain shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")
    print(f"Test shape: {test.shape}")
    return train, validation, test

# ------------------ 2-COVARIATE PIPELINE (NEW) ------------------ #

AGE_ALIASES  = ["Age", "Age Group", "Age Groups", "Five-Year Age Groups", "Ten-Year Age Groups"]
SEX_ALIASES  = ["Sex", "Gender"]

def _pick_col(df, aliases, must_contain=None):
    cols = list(df.columns)
    # exact match first
    for a in aliases:
        if a in df.columns:
            return a
    # fuzzy: contains token(s)
    if must_contain:
        toks = [t.lower() for t in (must_contain if isinstance(must_contain, (list,tuple)) else [must_contain])]
        for c in cols:
            lc = str(c).lower()
            if all(t in lc for t in toks):
                return c
    # last resort: first alias-like substring
    key = aliases[0].split()[0].lower()
    for c in cols:
        if key in str(c).lower():
            return c
    raise KeyError(f"None of {aliases} found. Available: {cols}")

def process_two_covariates(files, cov1_name, cov1_aliases, cov2_name, cov2_aliases, out_prefix):
    """Read 1–2 files, normalize to [Month, cov1, cov2, Deaths], interpolate per (cov1,cov2), save."""
    parts = []
    for f in files:
        print(f"Loading {f} …")
        df = pd.read_excel(f)

        # month
        month = parse_date_column(df)

        # covariates & deaths
        c1 = _pick_col(df, cov1_aliases)
        c2 = _pick_col(df, cov2_aliases)
        dcol = _find_deaths_column(df)

        tmp = pd.DataFrame({
            "Month": month,
            cov1_name: df[c1],
            cov2_name: df[c2],
            "Deaths": clean_deaths_column(df[dcol]),
        }).dropna(subset=["Month", cov1_name, cov2_name])
        parts.append(tmp)

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["Month", cov1_name, cov2_name]).sort_values(["Month", cov1_name, cov2_name])

    # interpolate within each combo
    print("Interpolating within each group combo …")
    df = (df.groupby([cov1_name, cov2_name], group_keys=False)
            .apply(lambda g: g.assign(
                Deaths=(g["Deaths"]
                        .interpolate("linear", limit_direction="both")
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                        .fillna(0))
            ))
          .sort_values(["Month", cov1_name, cov2_name])
          .reset_index(drop=True))

    # save long
    out_long = os.path.join(PROCESSED_DATA_DIR, f"{out_prefix}_long.csv")
    df.to_csv(out_long, index=False)

    # wide (flatten columns for CSV/ML)
    wide = df.pivot(index="Month", columns=[cov1_name, cov2_name], values="Deaths").sort_index()
    wide.columns = [f"{cov1_name}={a}|{cov2_name}={b}" for (a,b) in wide.columns.to_list()]
    wide = wide.sort_index(axis=1)
    wide = wide.interpolate("linear", limit_direction="both").fillna(method="ffill").fillna(method="bfill").fillna(0)

    out_wide = os.path.join(PROCESSED_DATA_DIR, f"{out_prefix}_wide.csv")
    wide.to_csv(out_wide)

    # splits
    train_end = "2019-01-01"
    val_end   = "2020-01-01"
    train = wide[wide.index < train_end]
    val   = wide[(wide.index >= train_end) & (wide.index < val_end)]
    test  = wide[wide.index >= val_end]

    train.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{out_prefix}_train.csv"))
    val.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{out_prefix}_val.csv"))
    test.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{out_prefix}_test.csv"))

    print("Saved:")
    print(f" - {out_long}")
    print(f" - {out_wide}")
    print(f" - {out_prefix}_train/val/test.csv")

# ------------------ MAIN PROCESSING ------------------ #

def main():
    """Main data processing pipeline"""

    # --- AGE-ONLY (1 covariate) ---
    # print("\n" + "="*60)
    # print("PROCESSING AGE-AGGREGATED DATA")
    # print("="*60)
    # age_data = load_and_combine_files(
    #     AGE_FILE_2018_2023,
    #     AGE_FILE_2010_2017,
    #     groupby_col='Age',
    #     groupby_aliases=['Age', 'Age Group']
    # )
    # age_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_aggregated_long.csv'), index=False)
    # age_wide = create_wide_format(age_data, groupby_col='Age')
    # age_train, age_val, age_test = create_train_val_test_split(age_wide)
    # age_wide.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_aggregated_wide.csv'))
    # age_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_train.csv'))
    # age_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_val.csv'))
    # age_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_test.csv'))

    # print("\nAge-aggregated data:")
    # print(age_wide.describe())

    # --- AGE × SEX (2 covariates) ---
    print("\n" + "="*60)
    print("PROCESSING AGE × SEX (TWO-COVARIATE) DATA")
    print("="*60)
    process_two_covariates(
        files=[AGE_SEX_FILE_2010_2017, AGE_SEX_FILE_2018_2023],
        cov1_name="Age",  cov1_aliases=AGE_ALIASES,
        cov2_name="Sex",  cov2_aliases=SEX_ALIASES,
        out_prefix="age_sex"
    )

if __name__ == "__main__":
    main()
