import streamlit as st
import pandas as pd
import json
import sql_db
from prompts.prompts import SYSTEM_MESSAGE
from azure_openai import get_completion_from_messages
import plotly.express as px
import os

# Load and apply the external CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
load_css("style.css")

@st.cache_data
def get_data(query, db_file):
    """Query the database and return results."""
    return sql_db.query_database(query, db_file)

@st.cache_data
def get_schema(db_file):
    """Get schema representation for the database."""
    return sql_db.get_schema_representation(db_file)

def save_temp_file(uploaded_file):
    """Save the uploaded file as a temporary database file."""
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

def generate_sql_query(user_message, table_name, schema):
    """Generate SQL query using the formatted system message."""
    formatted_system_message = SYSTEM_MESSAGE.format(
        table_name=table_name,
        schema=json.dumps(schema, indent=2)
    )

    response = get_completion_from_messages(formatted_system_message, user_message)
    return response

def handle_query_response(response, db_file):
    """Parse the response from the API and display the result."""
    try:
        json_response = json.loads(response)
        query = json_response.get('query', '')

        if query:
            st.markdown("<div class='success'>SQL Query generated successfully!</div>", unsafe_allow_html=True)
            st.write("Generated SQL Query:")
            st.code(query, language="sql")

            # Execute query and display results
            sql_results = get_data(query, db_file)
            if not sql_results.empty:
                st.write("Query Results:")
                st.dataframe(sql_results)

                # Visualization
                chart_type = st.selectbox("Choose Chart Type", ["None"] + sql_results.columns.tolist())
                if chart_type != "None":
                    fig = px.bar(sql_results, x=sql_results.columns[0], y=chart_type) if chart_type in sql_results.columns else None
                    if fig:
                        st.plotly_chart(fig)
                
                # Export and Reporting
                st.download_button(
                    label="Download Results",
                    data=sql_results.to_csv(index=False),
                    file_name='query_results.csv',
                    mime='text/csv'
                )
                
                # Report Summary
                st.write("Summary Report:")
                st.write(f"Number of rows: {len(sql_results)}")
                st.write(f"Columns: {', '.join(sql_results.columns)}")
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

# Streamlit app layout
st.markdown("<div class='header'>NLP2SQL ☁️</div>", unsafe_allow_html=True)
st.markdown("<div class='main'>", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Upload Database", type=["db", "sqlite", "sql"])

if uploaded_file is not None:
    # Save file temporarily
    db_file = save_temp_file(uploaded_file)

    # Get schema representation for the database
    schemas = get_schema(db_file)
    table_names = list(schemas.keys())

    if table_names:
        # Allow user to select a table
        selected_table = st.selectbox("Select a Table", options=table_names)

        if selected_table:
            # Display the schema of the selected table
            schema = schemas[selected_table]
            st.write(f"Schema for table `{selected_table}`:")
            st.json(schema)

            # User input for natural language message
            user_message = st.text_input("Enter your message:", "")

            if user_message:
                with st.spinner('Generating SQL query...'):
                    # Generate SQL query using the new function
                    response = generate_sql_query(user_message, selected_table, schema)

                    # Handle API response and display results
                    handle_query_response(response, db_file)
    else:
        st.info("No tables found in the database.")
else:
    st.info("Please upload a database file to get started.")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Made By Lohit Kolluri, Alok Kumar, Aditya Sinha and Mohammad Arshad Ali</div>", unsafe_allow_html=True)
