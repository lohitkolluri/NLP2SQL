import streamlit as st
import pandas as pd
import json
import altair as alt
import sql_db
from prompts.prompts import SYSTEM_MESSAGE
from azure_openai import get_completion_from_messages
from dotenv import load_dotenv
import os
import openai
import io
import xlsxwriter

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API configuration using environment variables
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- UI Configuration ---
st.set_page_config(
    page_icon="🗃️",
    page_title="Chat with Your DB",
    layout="centered"
)

# Load and apply the external CSS file
def load_css(file_name):
    """Load and apply custom CSS for the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
load_css("style.css")

# --- Caching and Utility Functions ---
@st.cache_data
def get_data(query, db_file):
    """Query the database and return results."""
    return sql_db.query_database(query, db_file)

@st.cache_resource
def get_schema(db_file):
    """Get schema representation for the database."""
    return sql_db.get_schema_representation(db_file)

def save_temp_file(uploaded_file):
    """Save the uploaded file as a temporary database file."""
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

def analyze_query(query):
    """Analyze the query for potential optimizations."""
    return query  # Return the original query for now

def generate_sql_query(user_message, table_name, schema):
    """Generate SQL query using the formatted system message."""
    formatted_system_message = SYSTEM_MESSAGE.format(
        table_name=table_name,
        schema=json.dumps(schema, indent=2)
    )
    response = get_completion_from_messages(formatted_system_message, user_message)
    return response

def create_chart(df, chart_type):
    """Create a chart based on the selected chart type."""
    chart_width = 600
    chart_height = 400

    base_chart = alt.Chart(df).properties(
        width=chart_width,
        height=chart_height
    ).configure_title(
        fontSize=18,
        fontWeight='bold',
        font='Roboto'
    )

    if chart_type == "Bar Chart":
        chart = base_chart.mark_bar().encode(
            x=alt.X(df.columns[0], title=df.columns[0]),
            y=alt.Y(df.columns[1], title=df.columns[1]),
            color=alt.Color(df.columns[1], legend=None)
        )

    elif chart_type == "Line Chart":
        chart = base_chart.mark_line().encode(
            x=alt.X(df.columns[0], title=df.columns[0]),
            y=alt.Y(df.columns[1], title=df.columns[1]),
            color=alt.Color(df.columns[1], legend=None)
        )

    elif chart_type == "Histogram":
        chart = base_chart.mark_bar().encode(
            alt.X(df.columns[0], bin=alt.Bin(maxbins=30), title=df.columns[0]),
            y='count()'
        )

    else:
        return None

    return chart

def handle_query_response(response, db_file):
    """Parse the response from the API and display results and charts."""
    try:
        json_response = json.loads(response)
        query = json_response.get('query', '')

        if not query:
            st.markdown("<div class='warning'><span class='material-icons icon'>warning</span>No query generated. Please refine your message.</div>", unsafe_allow_html=True)
            st.session_state.query_executed = False
            return

        st.markdown("<div class='success'><span class='material-icons icon'>check_circle</span>SQL Query generated successfully!</div>", unsafe_allow_html=True)
        st.write("Generated SQL Query:")
        st.code(query, language="sql")

        # Analyze and optimize the query (placeholder)
        optimized_query = analyze_query(query)

        # Execute query and display results
        sql_results = get_data(optimized_query, db_file)
        if sql_results.empty:
            st.markdown("<div class='warning'><span class='material-icons icon'>warning</span>The query returned no results.</div>", unsafe_allow_html=True)
            st.session_state.query_executed = False
            return

        st.write("Query Results:")
        st.dataframe(sql_results)

        # Store results in session state
        st.session_state.sql_results = sql_results
        st.session_state.query_executed = True
        
        if len(sql_results.columns) >= 2:
            # Visualization options
            st.sidebar.markdown("### Visualization Options")
            chart_type = st.sidebar.selectbox("Select Chart Type", options=["None", "Bar Chart", "Line Chart", "Histogram"])

            if chart_type != "None":
                chart = create_chart(sql_results, chart_type)
                if chart:
                    st.altair_chart(chart)

        # Export and Reporting
        export_format = st.selectbox("Select Export Format", options=["CSV", "JSON", "Excel"])
        if export_format == "CSV":
            st.download_button(
                label="Download Results as CSV",
                data=sql_results.to_csv(index=False),
                file_name='query_results.csv',
                mime='text/csv'
            )
        elif export_format == "JSON":
            st.download_button(
                label="Download Results as JSON",
                data=sql_results.to_json(orient='records'),
                file_name='query_results.json',
                mime='application/json'
            )
        elif export_format == "Excel":
            # Create an in-memory buffer for the Excel file
            output = io.BytesIO()
            
            # Use pd.ExcelWriter to write the DataFrame to the buffer
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                sql_results.to_excel(writer, index=False, sheet_name='Results')
            
            # Seek to the beginning of the stream
            output.seek(0)

            # Create the download button for the Excel file
            st.download_button(
                label="Download Results as Excel",
                data=output,
                file_name='query_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        # Report Summary
        st.write("Summary Report:")
        st.write(f"Number of rows: {len(sql_results)}")
        st.write(f"Columns: {', '.join(sql_results.columns)}")

        # Save query to history
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        st.session_state.query_history.append(query)

    except json.JSONDecodeError:
        st.markdown("<div class='error'><span class='material-icons icon'>error</span>Failed to decode the response. Please try again.</div>", unsafe_allow_html=True)
    except KeyError:
        st.markdown("<div class='error'><span class='material-icons icon'>error</span>Unexpected response format. Please check the system message and API response.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='error'><span class='material-icons icon'>error</span>An unexpected error occurred: {e}</div>", unsafe_allow_html=True)

# --- Streamlit App Layout ---
st.sidebar.markdown("<div class='header'><span class='material-icons icon'>data_usage</span>NLP2SQL</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload Database", type=["db", "sqlite", "sql"])

if uploaded_file is not None:
    db_file = save_temp_file(uploaded_file)

    # Get schema representation for the database
    schemas = get_schema(db_file)
    table_names = list(schemas.keys())

    if table_names:
        # Allow user to select a table
        selected_table = st.sidebar.selectbox("Table", options=table_names, format_func=lambda x: f"{x} 🗃")

        if selected_table:
            st.markdown(f"<div class='title'>Table: {selected_table} 🗄</div>", unsafe_allow_html=True)
            schema = schemas[selected_table]
            with st.expander("View Schema", expanded=True):
                st.json(schema)
               
            user_message = st.text_input("Enter your query message:", key="user_message")

            if user_message:
                with st.spinner('Generating SQL query...'):
                    response = generate_sql_query(user_message, selected_table, schema)
                    handle_query_response(response, db_file)
    else:
        st.info("No tables found in the database.")
else:
    st.info("Please upload a database file to get started.")

# Query history in collapsible sidebar
with st.sidebar.expander("Query History", expanded=False):
    if "query_history" in st.session_state and st.session_state.query_history:
        for i, past_query in enumerate(st.session_state.query_history, 1):
            if st.button(f"Re-run Query {i}"):
                user_message = past_query  # Set the user message to the selected query
                with st.spinner('Generating SQL query...'):
                    response = generate_sql_query(user_message, selected_table, schema)
                    handle_query_response(response, db_file)
            st.markdown(f"**Query {i}:**")
            st.code(past_query, language="sql")
    else:
        st.info("No query history available.")
