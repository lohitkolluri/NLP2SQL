import os  
import openai  
from dotenv import load_dotenv  

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API configuration using environment variables
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")  
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")  

def get_completion_from_messages(system_message, user_message, model="NLP2SQL", temperature=0, max_tokens=500) -> str:
    """
    Generate a completion response from OpenAI's API based on the given system and user messages.
    
    Parameters:
    - system_message (str): The system message for setting the assistant's behavior.
    - user_message (str): The user's message or query.
    - model (str): The name of the model to use for the completion (default is "NLP2SQL").
    - temperature (float): The sampling temperature (default is 0).
    - max_tokens (int): The maximum number of tokens to generate in the response (default is 500).
    
    Returns:
    - str: The content of the generated response.
    """
    
    # Create the messages list containing the system and user messages
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{user_message}"}
    ]
    
    # Generate a completion response from the OpenAI API
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    
    # Return the content of the generated response
    return response.choices[0].message["content"]
