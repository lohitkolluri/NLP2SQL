import os
import re
import json

import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from graphviz import Digraph
import sql_db

from sql_db import *
from prompts.prompts import SYSTEM_MESSAGE
from streamlit_extras.chart_container import chart_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
from azure_openai import get_completion_from_messages

st.set_page_config(page_icon="🗃️", page_title="Chat with Your DB", layout="centered")

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


@st.cache_data
def get_data(query: str, db_name: str, db_type: str, host: str = None, user: str = None, password: str = None) -> pd.DataFrame:
    """Fetch results from the database based on the provided SQL query."""
    return sql_db.query_database(query, db_name, db_type, host, user, password)


def save_temp_file(uploaded_file) -> str:
    """Save the uploaded database file temporarily."""
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path
    
def generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 3) -> str:
    """Generate SQL query using the provided message and schemas for all tables, handling ambiguity and explaining the path chosen."""
    formatted_system_message = SYSTEM_MESSAGE.format(
        schemas=json.dumps(schemas, indent=2)
    )
    decision_log = []
    paths_summary = []
    decision_flow = []
    for attempt in range(max_attempts):
        response = get_completion_from_messages(formatted_system_message, user_message)
        
        try:
            json_response = json.loads(response)
            query = json_response.get('query', None)
            error = json_response.get('error', None)
            paths_considered = json_response.get('paths_considered', [])
            final_choice = json_response.get('final_choice', '')
            tables_and_columns = json_response.get('tables_and_columns', [])

            # Log and track decisions for the flowchart
            decision_flow.append(f"Attempt {attempt + 1}: Received Response")

            if error:
                decision_log.append(f"**Attempt {attempt + 1}**: Error occurred: {error}. Retrying...")
                decision_flow.append(f"Error: {error}.")
                continue

            if query is None:
                decision_log.append(f"**Attempt {attempt + 1}**: No valid SQL query was generated. Retrying with more details...")
                decision_flow.append("No valid SQL query. Retry.")
                continue

            # Log paths considered by OpenAI (if available)
            if paths_considered:
                for path in paths_considered:
                    tables = ', '.join(path['tables'])
                    paths_summary.append(f"Path considered: {path['description']} involved the tables: {tables}.")
                    decision_log.append(f"**Attempt {attempt + 1}**: Path chosen - {path['description']} using tables {tables}.")
                    decision_flow.append(f"Path: {path['description']} using tables: {tables}")

            if tables_and_columns:
                # Log the tables and columns passed through
                for entry in tables_and_columns:
                    table = entry['table']
                    columns = ', '.join(entry['columns'])
                    decision_log.append(f"**Attempt {attempt + 1}**: Passed through table `{table}` with columns {columns}.")
                    decision_flow.append(f"Table: {table} | Columns: {columns}")

            # Log the entire path of tables for the final query
            final_tables = [entry['table'] for entry in tables_and_columns]
            decision_log.append(f"**Final Path of Tables**: {', '.join(final_tables)}.")

            if final_choice:
                decision_log.append(f"**Final Decision**: The final path chosen was: {final_choice}.")
                decision_flow.append(f"Final Path: {final_choice}")

            if validate_sql_query(query):
                decision_log.append(f"**Validation**: The generated SQL query was successfully validated.")
                decision_flow.append("Query validated successfully.")
                
                # Create natural language summary of the decision log
                natural_language_summary = get_natural_language_summary(query, paths_summary)
                decision_log.append("### Decision Log Summary:")
                decision_log.append(natural_language_summary)

                return json.dumps({
                    "query": query,
                    "decision_log": decision_log,
                    "decision_flow": decision_flow,
                })
            else:
                user_message += " Please ensure that the query adheres to valid SQL syntax."
                decision_flow.append("Invalid SQL syntax. Retry.")

        except json.JSONDecodeError:
            decision_log.append(f"**Attempt {attempt + 1}**: Failed to decode JSON. Retrying...")
            user_message += " The response was not valid JSON. Please provide additional clarity."
            decision_log.append(f"Raw response received: {response}")
            decision_flow.append(f"JSON decode error.")

    return json.dumps({
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": decision_log,  # Return the flow even on failure
    })


def get_natural_language_summary(query: str, paths_summary: list) -> str:
    """Generate a natural language summary using OpenAI based on the query and paths considered."""
    summary_prompt = (
        f"Given the SQL query: '{query}', the paths considered for generating this query were:\n"
        f"{' '.join(paths_summary)}\n"
        f"Please provide a concise natural language explanation of the decision-making process."
    )

    # Call OpenAI API to generate the natural language summary
    response = get_completion_from_messages(SYSTEM_MESSAGE, summary_prompt)
    
    # Provide a human-readable summary, or fallback if the response is empty
    return response.strip() if response else "Summary generation failed."


def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> alt.Chart:
    """Create a chart based on selected chart type and columns."""
    base_chart = alt.Chart(df).properties(width=600, height=400).configure_title(fontSize=18, fontWeight='bold', font='Roboto')

    if chart_type == "Bar Chart":
        chart = base_chart.mark_bar().encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(y_col, legend=None)
        )
    elif chart_type == "Line Chart":
        chart = base_chart.mark_line().encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(y_col, legend=None)
        )
    elif chart_type == "Scatter Plot":
        chart = base_chart.mark_circle().encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            tooltip=[x_col, y_col]
        )
    elif chart_type == "Area Chart":
        chart = base_chart.mark_area().encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            tooltip=[x_col, y_col]
        )
    elif chart_type == "Histogram":
        chart = base_chart.mark_bar().encode(
            alt.X(x_col, bin=alt.Bin(maxbins=30), title=x_col),
            y='count()'
        )
    else:
        return None

    return chart

def display_summary_statistics(df: pd.DataFrame) -> None:
    """Show summary statistics for the DataFrame."""
    st.subheader("Summary Statistics")
    st.write(df.describe())

def handle_query_response(response: str, db_name: str, db_type: str, host: str = None, user: str = None, password: str = None) -> None:
    """Process the API response and display query results, charts, and decision flow."""
    try:
        json_response = json.loads(response)
        query = json_response.get('query', '')
        error = json_response.get('error', '')
        decision_log = json_response.get('decision_log', [])

        if error:
            st.error(f"Error generating SQL query: {error}")
            return

        if not query:
            st.warning("No query generated. Please refine your message.")
            return

        st.success("SQL Query generated successfully!")
        st.code(query, language="sql")

        # Display decision log with paths and reasons
        if decision_log:
            st.subheader("Decision Log")
            for log in decision_log:
                st.write(log)

        sql_results = get_data(query, db_name, db_type, host, user, password)
        
        if sql_results.empty:
            st.warning("The query returned no results.")
            return

        # Check for duplicate column names
        if sql_results.columns.duplicated().any():
            st.error("The query returned a DataFrame with duplicate column names. Please modify your query to avoid this.")
            return

        # Convert object columns to datetime if possible
        for col in sql_results.select_dtypes(include=['object']):
            try:
                sql_results[col] = pd.to_datetime(sql_results[col])
            except ValueError:
                continue  # Skip columns that cannot be converted

        st.subheader("Query Results:")
        filtered_results = dataframe_explorer(sql_results, case=False)
        st.dataframe(filtered_results, use_container_width=True)

        display_summary_statistics(filtered_results)

        if len(filtered_results.columns) >= 2:
            st.sidebar.markdown("### Visualization Options")
            x_col = st.sidebar.selectbox("Select X-axis Column", options=filtered_results.columns)
            y_col = st.sidebar.selectbox("Select Y-axis Column", options=filtered_results.columns)
            chart_type = st.sidebar.selectbox("Select Chart Type", options=["None", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram"])
            
            if chart_type != "None" and x_col != y_col:
                chart = create_chart(filtered_results, chart_type, x_col, y_col)
                if chart:
                    with chart_container(data=filtered_results, export_formats=["CSV", "Parquet"]):
                        st.altair_chart(chart)

        export_format = st.selectbox("Select Export Format", options=["CSV", "Parquet"])
        export_results(filtered_results, export_format)

        st.write("Summary Report:")
        st.write(f"Number of rows: {len(filtered_results)}")
        st.write(f"Columns: {', '.join(filtered_results.columns)}")

        if "query_history" not in st.session_state:
            st.session_state.query_history = []
            st.session_state.query_timestamps = []

        st.session_state.query_history.append(query)
        st.session_state.query_timestamps.append(pd.Timestamp.now())

    except json.JSONDecodeError:
        st.error("Failed to decode the response. Please try again.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def generate_flowchart(decision_flow: list) -> Digraph:
    """Generate a flowchart based on the decision flow."""
    flowchart = Digraph(comment='Decision Flow')

    for index, step in enumerate(decision_flow):
        flowchart.node(str(index), step)

        # Create edges between consecutive steps
        if index > 0:
            flowchart.edge(str(index - 1), str(index))

    return flowchart


def validate_sql_query(query: str) -> bool:
    """Check the SQL query for validity and potentially harmful commands."""
    if not isinstance(query, str):
        return False
        
    # List of disallowed keywords (case-insensitive)
    disallowed = r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b'
        
    # Check for disallowed keywords
    if re.search(disallowed, query, re.IGNORECASE):
        return False
        
    # Basic syntax checks
    if not query.strip().lower().startswith(('select', 'with')):
        return False
        
    # Check for balanced parentheses
    if query.count('(') != query.count(')'):
        return False
        
    return True


def export_results(sql_results: pd.DataFrame, export_format: str) -> None:
    """Enable exporting of results in selected format."""
    if export_format == "CSV":
        st.download_button(
            label="Download Results as CSV",
            data=sql_results.to_csv(index=False),
            file_name='query_results.csv',
            mime='text/csv'
        )
    elif export_format == "Parquet":
        st.download_button(
            label="Download Results as Parquet",
            data=sql_results.to_parquet(index=False),
            file_name='query_results.parquet',
            mime='application/parquet'
        )
    else:
        st.error("Selected export format is not supported.")

        
# Streamlit App Layout
db_type = st.sidebar.selectbox("Select Database Type", options=["SQLite", "PostgreSQL"])
if db_type == "SQLite":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite Database", type=["db", "sqlite", "sql"])

    if uploaded_file is not None:
        db_file = save_temp_file(uploaded_file)
        schemas = get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if table_names:
            selected_tables = st.sidebar.multiselect("Select Tables", options=table_names, format_func=lambda x: f"{x} 🗃")
            if selected_tables:
                st.markdown(f"<div class='title'>Selected Tables: {', '.join(selected_tables)} 🗄</div>", unsafe_allow_html=True)
                for table in selected_tables:
                    with st.expander(f"View Schema: {table}", expanded=False):
                        st.json(schemas[table])

                user_message = st.text_input("Enter your query message:", key="user_message")
                if user_message:
                    with st.spinner('Generating SQL query...'):
                        response = generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                        handle_query_response(response, db_file, db_type='sqlite')

        else:
            st.info("No tables found in the database.")
    else:
        st.info("Please upload a database file to start.")

elif db_type == "PostgreSQL":
    postgres_host = st.sidebar.text_input("PostgreSQL Host")
    postgres_db = st.sidebar.text_input("Database Name")
    postgres_user = st.sidebar.text_input("Username")
    postgres_password = st.sidebar.text_input("Password", type="password")

    if postgres_host and postgres_db and postgres_user and postgres_password:
        schemas = get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            selected_tables = st.sidebar.multiselect("Select Tables", options=table_names, format_func=lambda x: f"{x} 🗃")
            if selected_tables:
                st.markdown(f"<div class='title'>Selected Tables: {', '.join(selected_tables)} 🗄</div>", unsafe_allow_html=True)
                for table in selected_tables:
                    with st.expander(f"View Schema: {table}", expanded=False):
                        st.json(schemas[table])

                user_message = st.text_input("Enter your query message:", key="user_message")
                if user_message:
                    with st.spinner('Generating SQL query...'):
                        response = generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                        handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)

        else:
            st.info("No tables found in the database.")
    else:
        st.info("Please fill in all PostgreSQL connection details to start.")

# Query history in collapsible sidebar
with st.sidebar.expander("Query History", expanded=False):
    if "query_history" in st.session_state and st.session_state.query_history:
        st.write("### Saved Queries")
        query_history_df = pd.DataFrame({
            "Query": st.session_state.query_history,
            "Timestamp": pd.to_datetime(st.session_state.query_timestamps)
        })

        # Display query history as a table
        st.dataframe(query_history_df, use_container_width=True)

        for i, (past_query, timestamp) in enumerate(zip(st.session_state.query_history, st.session_state.query_timestamps), 1):
            st.markdown(f"**Query {i}:** {past_query} \n*Executed on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
            if st.button(f"Re-run Query {i}", key=f"rerun_query_{i}"):
                user_message = past_query
                with st.spinner('Generating SQL query...'):
                    response = generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                    handle_query_response(response, db_file if db_type == "SQLite" else postgres_db, db_type, host=postgres_host if db_type == "PostgreSQL" else None, user=postgres_user if db_type == "PostgreSQL" else None, password=postgres_password if db_type == "PostgreSQL" else None)
    else:
        st.info("No query history available.")