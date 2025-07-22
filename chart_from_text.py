from llm_engine import ask_llm

def generate_chart_code(df, query: str) -> str:
    """
    Generates Python code to create a chart based on a natural language query using matplotlib or seaborn.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        query (str): A natural language description of the desired chart.

    Returns:
        str: Python code as a string (matplotlib/seaborn).
    """
    preview = df.head(5).to_csv(index=False)
    schema_info = "\n".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])

    prompt = f"""
You are a senior data visualization expert. 
The user wants to create a chart using matplotlib or seaborn from a Pandas DataFrame.

Dataset Schema:
{schema_info}

Dataset Preview:
{preview}

Task:
Generate Python code that creates this chart:
"{query}"

Constraints:
- Use only matplotlib and seaborn
- Assume 'df' is already loaded
- End the code with 'plt.show()'
- Return only the Python code block without any extra text or markdown
"""

    try:
        return ask_llm(prompt)
    except Exception:
        return "# Error: Could not generate chart code from query."
