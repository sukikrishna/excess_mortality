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

# ------------------ HELPER FUNCTIONS ------------------ #

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
    
    Parameters:
    - df: DataFrame with time series data
    - groupby_cols: list of columns to group by (e.g., ['State'] or ['Sex'])
    - value_col: name of the column to interpolate
    """
    df = df.copy()
    
    # Ensure the dataframe is sorted by date
    df = df.sort_values(['Month'] + groupby_cols).reset_index(drop=True)
    
    # Group by the specified columns and interpolate
    def interpolate_group(group):
        group = group.copy()
        # Linear interpolation for interior missing values
        group[value_col] = group[value_col].interpolate(method='linear', limit_direction='both')
        # Fill remaining NaN values (at boundaries) with forward/backward fill
        group[value_col] = group[value_col].fillna(method='ffill').fillna(method='bfill')
        # If still any NaN (entire group was NaN), fill with 0
        group[value_col] = group[value_col].fillna(0)
        return group
    
    if groupby_cols:
        df = df.groupby(groupby_cols, group_keys=False).apply(interpolate_group)
    else:
        df = interpolate_group(df)
    
    return df


def load_and_combine_files(file1, file2, groupby_col):
    """
    Load two Excel files, combine them, and return cleaned dataframe.
    
    Parameters:
    - file1: path to newer file (2018-2023)
    - file2: path to older file (2010-2017)
    - groupby_col: column name for grouping ('State' or 'Sex')
    """
    print(f"Loading {file1}...")
    df1 = pd.read_excel(file1)
    
    print(f"Loading {file2}...")
    df2 = pd.read_excel(file2)
    
    # Parse dates
    df1['Month'] = parse_date_column(df1)
    df2['Month'] = parse_date_column(df2)
    
    # Clean deaths column
    df1['Deaths'] = clean_deaths_column(df1['Deaths'])
    df2['Deaths'] = clean_deaths_column(df2['Deaths'])
    
    # Select relevant columns
    cols_to_keep = ['Month', groupby_col, 'Deaths']
    df1 = df1[cols_to_keep].dropna(subset=['Month', groupby_col])
    df2 = df2[cols_to_keep].dropna(subset=['Month', groupby_col])
    
    # Combine dataframes
    combined = pd.concat([df2, df1], ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    combined = combined.drop_duplicates(subset=['Month', groupby_col], keep='first')
    
    # Sort by date and group
    combined = combined.sort_values(['Month', groupby_col]).reset_index(drop=True)
    
    print(f"Combined data shape: {combined.shape}")
    print(f"Date range: {combined['Month'].min()} to {combined['Month'].max()}")
    print(f"Unique {groupby_col}: {combined[groupby_col].nunique()}")
    print(f"Missing values before interpolation: {combined['Deaths'].isna().sum()}")
    
    # Interpolate missing values
    combined = interpolate_missing_values(combined, groupby_cols=[groupby_col])
    
    print(f"Missing values after interpolation: {combined['Deaths'].isna().sum()}")
    
    return combined


def create_wide_format(df, groupby_col):
    """
    Convert long format to wide format for multivariate time series.
    Each column will be a different series (e.g., Female, Male or different states).
    
    Returns:
    - DataFrame with Month as index and each group as a column
    """
    # Pivot to wide format
    wide = df.pivot(index='Month', columns=groupby_col, values='Deaths')
    
    # Sort by date
    wide = wide.sort_index()
    
    # Fill any remaining NaN values with interpolation
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
    
    print("="*60)
    print("PROCESSING SEX-AGGREGATED DATA")
    print("="*60)
    
    # Process sex-aggregated data
    sex_data = load_and_combine_files(
        SEX_FILE_2018_2023, 
        SEX_FILE_2010_2017, 
        groupby_col='Sex'
    )
    
    # Save long format
    sex_data.to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'sex_aggregated_long.csv'), 
        index=False
    )
    
    # Create wide format
    sex_wide = create_wide_format(sex_data, groupby_col='Sex')
    
    # Create splits
    sex_train, sex_val, sex_test = create_train_val_test_split(sex_wide)
    
    # Save wide format and splits
    sex_wide.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_aggregated_wide.csv'))
    sex_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_train.csv'))
    sex_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_val.csv'))
    sex_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sex_test.csv'))
    
    print("\n" + "="*60)
    print("PROCESSING STATE-AGGREGATED DATA")
    print("="*60)
    
    # Process state-aggregated data
    state_data = load_and_combine_files(
        STATE_FILE_2018_2023,
        STATE_FILE_2010_2017,
        groupby_col='State'
    )
    
    # Save long format
    state_data.to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'state_aggregated_long.csv'),
        index=False
    )
    
    # Create wide format
    state_wide = create_wide_format(state_data, groupby_col='State')
    
    # Create splits
    state_train, state_val, state_test = create_train_val_test_split(state_wide)
    
    # Save wide format and splits
    state_wide.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_aggregated_wide.csv'))
    state_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_train.csv'))
    state_val.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_val.csv'))
    state_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'state_test.csv'))
    
    print("\n" + "="*60)
    print("DATA PROCESSING COMPLETE")
    print("="*60)
    print(f"\nAll processed files saved to: {PROCESSED_DATA_DIR}/")
    print("\nFiles created:")
    print("  Sex-aggregated:")
    print("    - sex_aggregated_long.csv")
    print("    - sex_aggregated_wide.csv")
    print("    - sex_train.csv, sex_val.csv, sex_test.csv")
    print("  State-aggregated:")
    print("    - state_aggregated_long.csv")
    print("    - state_aggregated_wide.csv")
    print("    - state_train.csv, state_val.csv, state_test.csv")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nSex-aggregated data:")
    print(sex_wide.describe())
    
    print("\nState-aggregated data (first 5 states):")
    print(state_wide.iloc[:, :5].describe())


if __name__ == "__main__":
    main()