import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    temperature: float = 0.3,
) -> str:
    try:
        combined_message = f"{system_message}\n\nUser Query: {user_message}"
        # print(f"Combined Message:\n{combined_message}")
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            contents=combined_message,
            generation_config={"temperature": temperature}
        )
        text = response.text if isinstance(response.text, str) else str(response.text)
        clean_text = re.sub(r'```json\n|\n```', '', text)
        return clean_text
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        return error_msg