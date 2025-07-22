import streamlit.components.v1 as components

# Embed GA tracker
components.iframe("https://yashu-20.github.io/Data-Copilot/ga_embed.html", height=0)

import time
start = time.time()

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Custom Modules
from data_cleaner import clean_data
from insights_writer import generate_insights, ask_custom_question
from dashboard_planner import suggest_dashboard_layout, generate_dashboard
from eda_module import plot_correlation_heatmap
from cluster_segment import perform_clustering
from sklearn.ensemble import RandomForestClassifier
from llm_engine import ask_llm
from ml_module import train_models
from predictor import predict_row
from report_generator import create_pdf_report
from chart_from_text import generate_chart_code
from anomaly_detector import detect_anomalies
from explainability import explain_model
from geo_mapper import plot_map

# App Settings
st.set_page_config(page_title="Data Copilot AI", layout="wide")
st.title("Data Copilot AI")
st.markdown("""
Upload any `.csv` dataset (e.g., e-commerce, HR, healthcare, sales).  
The app will clean, analyze, visualize, model, and answer your questions using GPT.
""")

# File Upload
use_local_folder = st.sidebar.checkbox("Use Local 'Data/' Folder")
combined_df = pd.DataFrame()

if use_local_folder:
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        st.warning("No CSV files found in 'Data/'.")
        st.stop()
    st.success(f"Loaded: {', '.join(csv_files)}")
    for file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, file), low_memory=False)
        time_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        combined_df = pd.concat([combined_df, df], ignore_index=True)
else:
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Please upload at least one CSV file.")
        st.stop()
    st.success(f"Loaded: {', '.join([f.name for f in uploaded_files])}")
    for file in uploaded_files:
        df = pd.read_csv(file, low_memory=False)
        time_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Main Logic
if not combined_df.empty:
    st.header("Data Cleaning & Preview")
    combined_df, cleaning_report = clean_data(combined_df)
    with st.expander(" Cleaning Report"):
        for k, v in cleaning_report.items():
            st.markdown(f"**{k}**: {v}")

    st.dataframe(combined_df.sample(min(1000, len(combined_df))))
    st.markdown("### Column Overview")
    for col in combined_df.columns:
        st.markdown(f"• **{col}** — {combined_df[col].dtype}")

    if st.checkbox("Show Descriptive Stats"):
        st.write(combined_df.describe(include='all').transpose())

    if st.checkbox("Show Correlation Heatmap"):
        plot_correlation_heatmap(combined_df)

    if st.checkbox("Show Geo Map"):
        plot_map(combined_df)

    if st.checkbox("Detect Anomalies"):
        combined_df = detect_anomalies(combined_df)
        st.dataframe(combined_df)

    st.header("Machine Learning & Explainability")
    target_col = st.selectbox("Select Target Column", combined_df.columns)
    model_container = {}

    if st.button("Train Models"):
        if combined_df[target_col].nunique() <= 1:
            st.error("Target column must have at least 2 unique values.")
            st.stop()
        if pd.api.types.is_datetime64_any_dtype(combined_df[target_col]):
            st.error("Target column cannot be a datetime.")
            st.stop()

        for col in combined_df.select_dtypes(include='datetime').columns:
            combined_df[col] = combined_df[col].view('int64')

        with st.spinner("Training in progress..."):
            try:
                result = train_models(combined_df, target_col)
                if isinstance(result, tuple) and len(result) == 2:
                    results, trained_models = train_models(combined_df, target_col, return_model=True)
                    model_container.update(trained_models)
                else:
                    results = result
                for model, report in results.items():
                    st.markdown(f"### {model}")
                    st.json(report)
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.button("Explain with SHAP") and 'Random Forest' in model_container:
        try:
            explain_model(model_container['Random Forest'], combined_df.drop(columns=[target_col]))
        except Exception as e:
            st.error(f"SHAP error: {e}")

    st.header("Meet DataNova – Your AI Data Companion")
    q = st.text_input("Ask anything about your data...")
    if st.button("Get GPT Answer") and q:
        try:
            st.markdown(ask_custom_question(combined_df, q))
        except Exception as e:
            st.error(f"LLM failed: {e}")
    
    st.header("Ask Anything")
    user_general_question = st.text_input("Ask anything from world knowledge, coding, history, science...", key="gen_q")

    if st.button("Ask DataNova"):
        if user_general_question:
            with st.spinner("Thinking..."):
                answer = ask_llm(user_general_question, system_role="You are an expert AI assistant.")
                st.markdown(answer)


    st.header("Natural Language Charting")
    chart_q = st.text_input("What chart do you want to see?")
    if st.button("Generate Chart") and chart_q:
        try:
            code = generate_chart_code(combined_df, chart_q)
            st.code(code)
            exec(code, globals())
        except Exception as e:
            st.error(f"Chart generation error: {e}")

    st.header("Dashboard Planner")
    dash_choice = st.radio("Choose:", ["Suggest Layout", "Auto Generate", "Skip"])
    if dash_choice == "Suggest Layout":
        st.markdown(suggest_dashboard_layout(combined_df))
    elif dash_choice == "Auto Generate":
        generate_dashboard(combined_df)

    st.header("GPT Insights")
    if st.button("Generate Insights"):
        try:
            st.markdown(generate_insights(combined_df))
        except Exception as e:
            st.error(f"Insight generation error: {e}")

    st.header("Export Report")
    if st.button("Export PDF Report"):
        create_pdf_report("report.pdf", "LLM Report", f"Auto-summary for {target_col}", {"Model": "Random Forest"})
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="llm_report.pdf")

    with st.expander("Optional: Clustering"):
        if st.button("Run Clustering"):
            fig = perform_clustering(combined_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No data loaded.")

st.success(f"Runtime: {round(time.time() - start, 2)}s")


import streamlit.components.v1 as components

# Google Analytics snippet (replace 'G-H863P6928M' with your own tracking ID if needed)
GA_TAG = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-H863P6928M"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-H863P6928M');
</script>
"""

# Inject into Streamlit (height=0 so it's invisible)
components.html(GA_TAG, height=0)


