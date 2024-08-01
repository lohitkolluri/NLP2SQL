from dotenv import load_dotenv
import os
import openai

# Load environment variables from the .env file
load_dotenv(dotenv_path='D:/Projects/NLP2SQL/.env')

# Retrieve API details from environment variables
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Print the key to verify it's loaded (ensure to remove or comment out in production)
print(f"Loaded API Key: {openai.api_key}")
print(f"Loaded API Endpoint: {openai.api_base}")

# Check if the API key and endpoint are loaded
if openai.api_key is None:
    raise ValueError("API Key not found. Please check your .env file and ensure OPENAI_API_KEY is set.")
if openai.api_base is None:
    raise ValueError("API Endpoint not found. Please check your .env file and ensure OPENAI_ENDPOINT is set.")


def get_completion_from_messages(system_message, user_message, model="NLP2SQL", temperature=0, max_tokens=500) -> str:
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message["content"]


if __name__ == "__main__":
    system_message = "You are a helpful assistant"
    user_message = "Hello, how are you?"
    print(get_completion_from_messages(system_message, user_message))
