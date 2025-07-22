import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_correlation_heatmap(df: pd.DataFrame, show: bool = True, title: str = "🔗 Correlation Heatmap", cmap: str = "coolwarm"):
    """
    Plots a correlation heatmap for numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        show (bool): If True, display the plot in Streamlit.
        title (str): Title for the heatmap.
        cmap (str): Color map to use for the heatmap.

    Returns:
        matplotlib.figure.Figure or None
    """
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numerical columns for correlation heatmap.")
        return None

    try:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title(title)

        if show:
            st.pyplot(fig)

        return fig
    except Exception as e:
        st.error(f" Failed to generate correlation heatmap: {e}")
        return None
