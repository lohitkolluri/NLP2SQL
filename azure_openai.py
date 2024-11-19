import os
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "OPENAI_API_KEY",
    "MODEL_NAME"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Initialize AzureOpenAI client
try:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info("AzureOpenAI client initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize AzureOpenAI client.")
    raise e

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024
) -> str:
    """
    Generate a completion response from OpenAI's API based on the given system and user messages.

    Parameters:
    - system_message (str): The system message for setting the assistant's behavior.
    - user_message (str): The user's message or query.
    - model (str): The name of the model to use for the completion (default from environment).
    - temperature (float): The sampling temperature (default is 1.0).
    - top_p (float): The top-p sampling parameter (default is 1.0).
    - max_tokens (int): The maximum number of tokens to generate in the response (default is 150).

    Returns:
    - str: The content of the generated response.
    """
    # Use environment variable for model if not provided
    if model is None:
        model = os.getenv("MODEL_NAME")

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        completion = response.choices[0].message.content
        logger.info("Received completion from OpenAI API.")
        return completion
    except Exception as e:
        logger.exception("Error while fetching completion from OpenAI API.")
        raise e
