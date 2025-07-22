from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def detect_anomalies(df, contamination=0.02, add_scores=False):
    """
    Detect anomalies using Isolation Forest on numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        contamination (float): The proportion of expected anomalies (default=0.02).
        add_scores (bool): Whether to add anomaly scores to the DataFrame.

    Returns:
        pd.DataFrame: Original DataFrame with an 'anomaly' column (True/False),
                      and optionally 'anomaly_score'.
    """
    numeric_df = df.select_dtypes(include='number').copy()

    if numeric_df.empty:
        df['anomaly'] = False
        return df

    # Fill NaNs with column medians for modeling
    numeric_df.fillna(numeric_df.median(), inplace=True)

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)

    # Add results to original DataFrame
    df = df.copy()  # To avoid SettingWithCopyWarning
    df['anomaly'] = preds == -1

    if add_scores:
        scores = model.decision_function(numeric_df)
        df['anomaly_score'] = scores

    return df
