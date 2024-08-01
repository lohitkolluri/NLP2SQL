import os
import json
import pandas as pd
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set OpenAI API configuration
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_wikisql_data(file_path):
    """Load WikiSQL data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def get_completion_from_messages(system_message, user_message, model="NLP2SQL", temperature=0, max_tokens=500):
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message["content"]


def generate_sql_query(system_message, user_message):
    response = get_completion_from_messages(system_message, user_message)
    return response


if __name__ == "__main__":
    # Load the WikiSQL training data
    train_data = load_wikisql_data('data/train.jsonl')

    # Example of using a single example from the dataset
    example = train_data.iloc[0]
    table_id = example['table_id']
    question = example['question']
    sql_query = example['sql']

    # Assuming the schema is stored in a separate file or object
    with open(f"data/{table_id}.json") as f:
        table_schema = json.load(f)

    system_message = f"""You are an AI assistant that converts natural language into a properly formatted SQL query. The database you are querying could be any SQL-based system (e.g., SQLite, MySQL, PostgreSQL, SQL Server). 

The table you will be querying is named "{table_schema['name']}". Here is the schema of the table:
{json.dumps(table_schema['schema'], indent=2)}

Please ensure the SQL query is compatible with the specified database system, but do not assume any specific SQL dialect or syntax. For example, avoid using database-specific functions or clauses that are not supported across different SQL systems.

Your output must be in JSON format with the following key-value pairs:
- "query": the SQL query that you generated
- "error": an error message if the query is invalid, or null if the query is valid

If you encounter any issues generating the query, provide a helpful error message in the "error" field."""

    user_message = question
    response = generate_sql_query(system_message, user_message)
    print(response)
