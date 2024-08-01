import streamlit as st
import pandas as pd
import json
import sql_db
from prompts.prompts import SYSTEM_MESSAGE
from azure_openai import get_completion_from_messages

# Custom CSS for dark theme styling
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #f1f1f1;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle, .instruction, .warning, .error, .success {
        color: #f1f1f1;
    }
    .warning {
        background-color: #5a5a5a;
        padding: 10px;
        border-radius: 5px;
    }
    .error {
        background-color: #ff4b4b;
        padding: 10px;
        border-radius: 5px;
    }
    .success {
        background-color: #4caf50;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput input {
        color: #f1f1f1;
        background-color: #333333;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput label {
        color: #f1f1f1;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>SQL Query Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload your database file or select an existing one.</p>", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Upload SQLite Database", type=["db", "sqlite"])

# If a file is uploaded, save it temporarily and create a connection
if uploaded_file is not None:
    with open("temp_database.db", "wb") as f:
        f.write(uploaded_file.read())
    
    # Use the uploaded database file
    db_file = "temp_database.db"

    # Get schema representation for the system message
    schemas = sql_db.get_schema_representation(db_file)

    # User input for natural language message
    user_message = st.text_input("Enter your message:", "")

    if user_message:
        with st.spinner('Generating SQL query...'):
            # Format system message with schema information
            formatted_system_message = SYSTEM_MESSAGE.format(schema=schemas.get('finances', {}))

            # Generate SQL query using Azure OpenAI
            response = get_completion_from_messages(formatted_system_message, user_message)

            try:
                # Parse JSON response and extract SQL query
                json_response = json.loads(response)
                query = json_response.get('query', '')

                if query:
                    st.markdown("<div class='success'>SQL Query generated successfully!</div>", unsafe_allow_html=True)
                    st.write("Generated SQL Query:")
                    st.code(query, language="sql")

                    # Execute query and display results
                    sql_results = sql_db.query_database(query, db_file)
                    if not sql_results.empty:
                        st.write("Query Results:")
                        st.dataframe(sql_results)
                    else:
                        st.markdown("<div class='warning'>The query returned no results.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning'>No query generated. Please refine your message.</div>", unsafe_allow_html=True)

            except json.JSONDecodeError:
                st.markdown("<div class='error'>Failed to decode the response. Please try again.</div>", unsafe_allow_html=True)
            except KeyError:
                st.markdown("<div class='error'>Unexpected response format. Please check the system message and API response.</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div class='error'>An unexpected error occurred: {e}</div>", unsafe_allow_html=True)

else:
    st.info("Please upload a SQLite database file to get started.")

st.markdown("</div>", unsafe_allow_html=True)
