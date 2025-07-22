from llm_engine import ask_llm
import pandas as pd


def generate_insights(df: pd.DataFrame, max_rows: int = 3) -> str:
    """
    Generates business insights using LLM from a small sample of the dataset.

    Args:
        df (pd.DataFrame): The full dataset.
        max_rows (int): Number of sample rows to include.

    Returns:
        str: Insights from the LLM.
    """
    if df.empty:
        return "⚠️ Dataset is empty. Cannot generate insights."

    selected_cols = df.select_dtypes(exclude=["object"]).columns[:10].tolist()
    sample_df = df[selected_cols].dropna().sample(n=min(max_rows, len(df)), random_state=42)

    if sample_df.empty:
        return "⚠️ Insufficient clean data to generate insights."

    sample_text = sample_df.to_markdown(index=False)[:1800]

    prompt = f"""
You are a senior business intelligence analyst.

Using the following dataset sample, generate:
1. A brief overview of the business performance.
2. Two important data trends or anomalies.
3. Three strategic recommendations that a business leader can act on.

### Data Sample:
{sample_text}
"""
    try:
        return ask_llm(prompt)
    except Exception as e:
        return f"❌ LLM Error during insight generation: {e}"


def ask_custom_question(df: pd.DataFrame, question: str, max_rows: int = 3) -> str:
    """
    Answers a custom question based on dataset sample using LLM.

    Args:
        df (pd.DataFrame): The full dataset.
        question (str): The user's question.
        max_rows (int): Number of rows to include in sample.

    Returns:
        str: LLM-generated answer.
    """
    if df.empty:
        return "⚠️ Dataset is empty. Cannot answer question."

    selected_cols = df.columns[:10].tolist()
    sample_df = df[selected_cols].dropna().sample(n=min(max_rows, len(df)), random_state=1)

    if sample_df.empty:
        return "⚠️ Insufficient clean data to answer the question."

    sample_text = sample_df.to_markdown(index=False)[:1800]

    prompt = f"""
You are a professional data analyst working with the following dataset sample.

### Dataset Sample:
{sample_text}

### User's Question:
{question}

Provide a clear and concise answer based only on the dataset.
"""
    try:
        return ask_llm(prompt)
    except Exception as e:
        return f"❌ LLM Error while answering the question: {e}"
