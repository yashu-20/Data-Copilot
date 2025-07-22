import shap
import streamlit as st
import pandas as pd

def explain_model(model, X, use_tree_explainer=True):
    """
    Display SHAP summary plots for model explainability in Streamlit.

    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Feature dataset.
        use_tree_explainer (bool): Use TreeExplainer if applicable (faster for tree models).
    """
    try:
        st.subheader("🔍 Model Explainability with SHAP")

        # Choose appropriate explainer
        if use_tree_explainer and hasattr(model, "predict"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X)

        # Compute SHAP values
        shap_values = explainer(X)

        # Plot SHAP summary
        st.set_option('deprecation.showPyplotGlobalUse', False)

        with st.expander("Beeswarm Plot (Feature Impact)"):
            shap.plots.beeswarm(shap_values)
            st.pyplot()

        with st.expander("Bar Plot (Mean Absolute SHAP Values)"):
            shap.plots.bar(shap_values)
            st.pyplot()

    except Exception as e:
        st.error(f"Error generating SHAP explanations: {e}")
