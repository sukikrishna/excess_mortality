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

# NEW: File paths for age-aggregated data
AGE_FILE_2018_2023 = os.path.join(RAW_DATA_DIR, 'Agg_Age_Year_Month.xls.xlsx')   # user upload
AGE_FILE_2010_2017 = os.path.join(RAW_DATA_DIR, 'Agg_Age_Year_Month_2017.xlsx')  # user upload

# ------------------ HELPER FUNCTIONS ------------------ #

# --- add/replace these helpers ---

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
    # Fuzzy/contains match fallback
    wanted_tokens = set(_normalize_colname(preferred_name).split())
    for norm, orig in norm_map.items():
        if 'age' in norm and 'year' not in norm:  # avoid catching Year
            # prefer names that also include 'group'
            if 'group' in norm:
                return orig
    # last resort: any column with 'age'
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
        # Extract month number from Month Code if it's a date
        month_vals = pd.to_datetime(df['Month Code'], errors='coerce').dt.month
        if month_vals.notna().sum() == 0:
            # Month Code might be numeric already
            month_vals = pd.to_numeric(df['Month Code'], errors='coerce')
        
        year_vals = pd.to_numeric(df['Year Code'], errors='coerce')
        
        # Create dates
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

# NEW: helper to standardize a grouping column (e.g., 'Age' vs 'Age Group')
def _standardize_group_column(df, candidates, new_name):
    """
    Ensure the dataframe has a single grouping column named `new_name`.
    Will use the first found candidate from `candidates`.
    """
    for cand in candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: new_name})
            return df
    # If none found, keep as-is (will error later in selection)
    return df


# def load_and_combine_files(file1, file2, groupby_col, groupby_aliases=None):
#     """
#     Load two Excel files, combine them, and return cleaned dataframe.
    
#     Parameters:
#     - file1: path to newer file (2018-2023)
#     - file2: path to older file (2010-2017)
#     - groupby_col: canonical column name for grouping ('State', 'Sex', or 'Age')
#     - groupby_aliases: optional list of alternative column names to map to `groupby_col`
#     """
#     print(f"Loading {file1}...")
#     df1 = pd.read_excel(file1)
#     print(f"Loading {file2}...")
#     df2 = pd.read_excel(file2)

#     # Standardize grouping column if aliases provided
#     if groupby_aliases:
#         df1 = _standardize_group_column(df1, groupby_aliases + [groupby_col], groupby_col)
#         df2 = _standardize_group_column(df2, groupby_aliases + [groupby_col], groupby_col)

#     # Parse dates
#     df1['Month'] = parse_date_column(df1)
#     df2['Month'] = parse_date_column(df2)
    
#     # Clean deaths column (handle alt names just in case)
#     death_col_1 = 'Deaths' if 'Deaths' in df1.columns else [c for c in df1.columns if c.lower() == 'deaths'][0]
#     death_col_2 = 'Deaths' if 'Deaths' in df2.columns else [c for c in df2.columns if c.lower() == 'deaths'][0]
#     df1['Deaths'] = clean_deaths_column(df1[death_col_1])
#     df2['Deaths'] = clean_deaths_column(df2[death_col_2])
    
#     # Select relevant columns
#     cols_to_keep = ['Month', groupby_col, 'Deaths']
#     df1 = df1[cols_to_keep].dropna(subset=['Month', groupby_col])
#     df2 = df2[cols_to_keep].dropna(subset=['Month', groupby_col])
    
#     # Combine dataframes
#     combined = pd.concat([df2, df1], ignore_index=True)
    
#     # Remove duplicates (keep first occurrence)
#     combined = combined.drop_duplicates(subset=['Month', groupby_col], keep='first')
    
#     # Sort by date and group
#     combined = combined.sort_values(['Month', groupby_col]).reset_index(drop=True)
    
#     print(f"Combined data shape: {combined.shape}")
#     print(f"Date range: {combined['Month'].min()} to {combined['Month'].max()}")
#     print(f"Unique {groupby_col}: {combined[groupby_col].nunique()}")
#     print(f"Missing values before interpolation: {combined['Deaths'].isna().sum()}")
    
#     # Interpolate missing values
#     combined = interpolate_missing_values(combined, groupby_cols=[groupby_col])
    
#     print(f"Missing values after interpolation: {combined['Deaths'].isna().sum()}")
    
#     return combined


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

    # Identify the true grouping column name in each file
    # Include a broad set of likely age field names for WONDER
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
    # If this raises again, you'll see the printed columns above
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
    """
    Convert long format to wide format for multivariate time series.
    """
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


# ------------------ MAIN PROCESSING ------------------ #

def main():
    """Main data processing pipeline"""
#     print("="*60)
#     print("PROCESSING SEX-AGGREGATED DATA")
#     print("="*60)
#     sex_data = load_and_combine_files(
#         SEX_FILE_2018_2023, 
#         SEX_FILE_2010_2017, 
#         groupby_col='Sex',
#         groupby_aliases=['Sex']  # explicit for clarity
#     )
#     sex_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_aggregated_long.csv'), index=False)
#     sex_wide = create_wide_format(sex_data, groupby_col='Sex')
#     sex_train, sex_val, sex_test = create_train_val_test_split(sex_wide)
#     sex_wide.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_aggregated_wide.csv'))
#     sex_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_train.csv'))
#     sex_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_val.csv'))
#     sex_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_test.csv'))
    
#     print("\n" + "="*60)
#     print("PROCESSING STATE-AGGREGATED DATA")
#     print("="*60)
#     state_data = load_and_combine_files(
#         STATE_FILE_2018_2023,
#         STATE_FILE_2010_2017,
#         groupby_col='State',
#         groupby_aliases=['State']  # explicit for clarity
#     )
#     state_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_aggregated_long.csv'), index=False)
#     state_wide = create_wide_format(state_data, groupby_col='State')
#     state_train, state_val, state_test = create_train_val_test_split(state_wide)
#     state_wide.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_aggregated_wide.csv'))
#     state_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_train.csv'))
#     state_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_val.csv'))
#     state_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_test.csv'))

    # NEW: AGE-AGGREGATED DATA
    print("\n" + "="*60)
    print("PROCESSING AGE-AGGREGATED DATA")
    print("="*60)
    # Handle potential column name variations like 'Age' or 'Age Group'
    age_data = load_and_combine_files(
        AGE_FILE_2018_2023,
        AGE_FILE_2010_2017,
        groupby_col='Age',                   # canonical name in the outputs
        groupby_aliases=['Age', 'Age Group'] # likely variants in the raw files
    )
    age_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_aggregated_long.csv'), index=False)
    age_wide = create_wide_format(age_data, groupby_col='Age')
    age_train, age_val, age_test = create_train_val_test_split(age_wide)
    age_wide.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_aggregated_wide.csv'))
    age_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_train.csv'))
    age_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_val.csv'))
    age_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'age_test.csv'))

#     print("\n" + "="*60)
#     print("DATA PROCESSING COMPLETE")
#     print("="*60)
#     print(f"\nAll processed files saved to: {PROCESSED_DATA_DIR}/")
#     print("\nFiles created:")
#     print("  Sex-aggregated:")
#     print("    - sex_aggregated_long.csv")
#     print("    - sex_aggregated_wide.csv")
#     print("    - sex_train.csv, sex_val.csv, sex_test.csv")
#     print("  State-aggregated:")
#     print("    - state_aggregated_long.csv")
#     print("    - state_aggregated_wide.csv")
#     print("    - state_train.csv, state_val.csv, state_test.csv")
#     print("  Age-aggregated:")
#     print("    - age_aggregated_long.csv")
#     print("    - age_aggregated_wide.csv")
#     print("    - age_train.csv, age_val.csv, age_test.csv")
    
#     print("\n" + "="*60)
#     print("SUMMARY STATISTICS")
#     print("="*60)
#     print("\nSex-aggregated data:")
#     print(sex_wide.describe())
#     print("\nState-aggregated data (first 5 states):")
#     print(state_wide.iloc[:, :5].describe())
    print("\nAge-aggregated data:")
    print(age_wide.describe())

if __name__ == "__main__":
    main()
