import streamlit as st
import plotly.express as px
import pandas as pd

def suggest_dashboard_layout(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()

    layout = "### Suggested Power BI–Style Dashboard Layout\n\n"
    layout += "- **KPIs**: Mean, Total, Min, Max of key numeric variables\n"
    layout += "- **Time Series**: Line chart of metric over date/time column\n"
    layout += "- **Category Analysis**: Bar chart / box plot of metric by category\n"
    layout += "- **Trends**: Correlation heatmap, scatter matrix for insights\n"
    layout += "- **Comparison**: Heatmap of 2 categorical fields (counts)\n"
    return layout

def generate_dashboard(df):
    st.markdown("##  Power BI–Style Auto Dashboard")

    df_small = df.copy()
    if len(df_small) > 10000:
        df_small = df_small.sample(10000, random_state=42)

    numeric_cols = df_small.select_dtypes(include="number").columns.tolist()
    cat_cols = df_small.select_dtypes(include="object").columns.tolist()
    datetime_cols = df_small.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()

    # Optional filter
    if cat_cols:
        selected_category = st.selectbox(" Filter by category (optional):", [None] + cat_cols)
        if selected_category:
            selected_value = st.selectbox(f"Select value from `{selected_category}`", df_small[selected_category].unique())
            df_small = df_small[df_small[selected_category] == selected_value]

    # KPIs
    st.markdown("###  Key Performance Indicators")
    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Avg {numeric_cols[0]}", f"{df_small[numeric_cols[0]].mean():,.2f}")
        col2.metric(f"Min {numeric_cols[0]}", f"{df_small[numeric_cols[0]].min():,.2f}")
        col3.metric(f"Max {numeric_cols[0]}", f"{df_small[numeric_cols[0]].max():,.2f}")

    if len(numeric_cols) > 1:
        st.metric(f"Total {numeric_cols[1]}", f"{df_small[numeric_cols[1]].sum():,.2f}")

    # Time Series
    if datetime_cols and numeric_cols:
        st.markdown("###  Time Series Trend")
        try:
            fig1 = px.line(df_small.sort_values(datetime_cols[0]), x=datetime_cols[0], y=numeric_cols[0],
                           title=f"{numeric_cols[0]} Over Time")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.warning(f"Time series skipped: {e}")

    # Bar Chart
    if cat_cols and numeric_cols:
        st.markdown(f"###  Avg {numeric_cols[0]} by {cat_cols[0]}")
        try:
            grouped = df_small.groupby(cat_cols[0])[numeric_cols[0]].mean().reset_index()
            fig2 = px.bar(grouped.sort_values(numeric_cols[0], ascending=False).head(10),
                          x=cat_cols[0], y=numeric_cols[0],
                          color=numeric_cols[0],
                          title=f"Top Categories by Avg {numeric_cols[0]}")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Bar chart skipped: {e}")

    # Heatmap (Cat vs Cat)
    if len(cat_cols) >= 2:
        st.markdown(f"###  Count Heatmap: {cat_cols[0]} vs {cat_cols[1]}")
        try:
            heat_df = df_small.groupby([cat_cols[0], cat_cols[1]]).size().reset_index(name="count")
            fig3 = px.density_heatmap(heat_df, x=cat_cols[0], y=cat_cols[1], z="count",
                                      color_continuous_scale="Viridis",
                                      title=f"{cat_cols[0]} vs {cat_cols[1]} Heatmap")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap skipped: {e}")

    # Box Plot
    if cat_cols and numeric_cols:
        st.markdown(f"###  Distribution: {numeric_cols[0]} by {cat_cols[0]}")
        try:
            fig4 = px.box(df_small, x=cat_cols[0], y=numeric_cols[0], points="all",
                          title=f"{numeric_cols[0]} Distribution by {cat_cols[0]}")
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.warning(f"Box plot skipped: {e}")

    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        st.markdown("###  Numeric Correlation")
        try:
            corr_matrix = df_small[numeric_cols].corr().round(2)
            fig5 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", 
                             title="Correlation Heatmap")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            st.warning(f"Correlation heatmap skipped: {e}")

    # Scatter Matrix
    if len(numeric_cols) >= 3:
        st.markdown("###  Numeric Feature Scatter Matrix")
        try:
            fig6 = px.scatter_matrix(df_small[numeric_cols[:5]], title="Scatter Matrix")
            st.plotly_chart(fig6, use_container_width=True)
        except Exception as e:
            st.warning(f"Scatter matrix skipped: {e}")
