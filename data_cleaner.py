import pandas as pd
import numpy as np
import re

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, replace non-alphanumeric with underscores.
    """
    df.columns = [
        re.sub(r'[^\w]+', '_', col.strip().lower()) for col in df.columns
    ]
    return df

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans the DataFrame by handling duplicates, missing values, constant columns,
    empty columns, and auto-converting numeric-looking strings.

    Returns:
        df (pd.DataFrame): Cleaned DataFrame
        report (dict): Summary of cleaning actions
    """
    report = {}

    # Step 1: Clean column names
    df = clean_column_names(df)
    report['column_names_cleaned'] = True

    # Step 2: Remove duplicates
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    after_rows = df.shape[0]
    report["duplicates_removed"] = before_rows - after_rows

    # Step 3: Drop empty or constant columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    df = df.drop(columns=empty_cols + constant_cols)
    report["empty_columns_removed"] = empty_cols
    report["constant_columns_removed"] = constant_cols

    # Step 4: Handle missing values
    missing_info = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        missing_pct = (missing_count / len(df)) * 100

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            missing_info[col] = f"Filled with median ({missing_pct:.1f}%)"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(method='ffill')
            missing_info[col] = f"Forward-filled dates ({missing_pct:.1f}%)"
        else:
            df[col] = df[col].fillna("Unknown")
            missing_info[col] = f"Filled with 'Unknown' ({missing_pct:.1f}%)"

    report["missing_values_handled"] = missing_info

    # Step 5: Convert object types that look like numbers
    numeric_converted_cols = []
    for col in df.select_dtypes(include='object'):
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notnull().sum() > 0.9 * len(converted):
                df[col] = converted
                numeric_converted_cols.append(col)
        except Exception as e:
            continue

    if numeric_converted_cols:
        report["converted_to_numeric"] = numeric_converted_cols

    return df, report
