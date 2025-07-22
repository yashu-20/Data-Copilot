import os
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
api_base = "https://api.groq.com/openai/v1"

# Validate API key
if not api_key:
    raise EnvironmentError("GROQ_API_KEY not found in environment or .env file!")

# Configure OpenAI client
openai.api_key = api_key
openai.api_base = api_base
client = openai.OpenAI(api_key=api_key, base_url=api_base)


def ask_llm(prompt: str, 
            system_role: str = "You are a professional data analyst.", 
            temperature: float = 0.7, 
            max_tokens: int = 1024) -> str:
    """
    Send a prompt to the Groq-hosted LLaMA3-70B model and return a response.

    Args:
        prompt (str): User message or instruction.
        system_role (str): Role definition for the assistant.
        temperature (float): Sampling temperature for creativity (0 = deterministic).
        max_tokens (int): Maximum tokens to return.

    Returns:
        str: Model's response or an error message.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30  # Optional: safety timeout in seconds
        )

        content = response.choices[0].message.content.strip()
        return content if content else "⚠️ Empty response received from the model."

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"
