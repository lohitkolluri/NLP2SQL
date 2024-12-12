import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import io
import json
import re
import logging
from typing import Optional
import pandas as pd
import altair as alt
import streamlit as st
import streamlit_nested_layout
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
import numpy as np
from streamlit_extras.chart_container import chart_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
import src.database.DB_Config as DB_Config
from src.prompts.Base_Prompt import SYSTEM_MESSAGE
from src.api.OpenAI_Config import get_completion_from_messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {
    "Bar Chart": "A chart that presents categorical data with rectangular bars.",
    "Line Chart": "A chart that displays information as a series of data points called 'markers' connected by straight line segments.",
    "Scatter Plot": "A plot that displays values for typically two variables for a set of data.",
    "Area Chart": "A chart that displays quantitative data visually, using the area below the line.",
    "Histogram": "A graphical representation of the distribution of numerical data."
}

# Page Configuration
st.set_page_config(
    page_icon="🗃️",
    page_title="Transforming Questions into Queries",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def load_system_message(schemas: dict) -> str:
    return SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))


@st.cache_data
def get_data(query: str, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch results from the database based on the provided SQL query.
    """
    return DB_Config.query_database(query, db_name, db_type, host, user, password)


def save_temp_file(uploaded_file) -> str:
    """
    Save the uploaded database file temporarily.
    """
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path


@st.cache_data(show_spinner=False)
def cached_generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 3) -> dict:
    return generate_sql_query(user_message, schemas, max_attempts)


def generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 1) -> dict:
    """
    Generate SQL query from user input with retry mechanism.
    """
    formatted_system_message = load_system_message(schemas)
    decision_log_sections = {
        "Query Input Details": [],
        "Preprocessing Steps": [],
        "Path Identification": [],
        "Ambiguity Detection": [],
        "Resolution Criteria": [],
        "Chosen Path Explanation": [],
        "Generated SQL Query": [],
        "Alternative Paths": [],
        "Execution Feedback": []
    }
    visualization_recommendation = None

    # Query Input Details
    decision_log_sections["Query Input Details"].append(
        f"User's input: `{user_message}`"
    )

    # Preprocessing Steps
    decision_log_sections["Preprocessing Steps"].append(
        "Normalized the input to lowercase and removed unnecessary whitespace for clarity."
    )

    for attempt in range(max_attempts):
        response = get_completion_from_messages(formatted_system_message, user_message)

        try:
            json_response = json.loads(response)
            query = json_response.get('query')
            error = json_response.get('error')
            paths_considered = json_response.get('paths_considered', [])
            final_choice = json_response.get('final_choice', '')
            tables_and_columns = json_response.get('tables_and_columns', [])
            visualization_recommendation = json_response.get('visualization_recommendation')

            if error:
                decision_log_sections["Execution Feedback"].append(
                    f"Attempt {attempt + 1}: We encountered an issue: {error}. We're trying again."
                )
                continue

            if not query:
                decision_log_sections["Execution Feedback"].append(
                    f"Attempt {attempt + 1}: We couldn't generate a valid SQL query. Let's try that again."
                )
                continue

            if paths_considered:
                decision_log_sections["Path Identification"].append(
                    "We identified multiple potential paths to generate the query:"
                )
                for idx, path in enumerate(paths_considered, start=1):
                    tables = path['tables']
                    columns = [col for sublist in path['columns'] for col in sublist]
                    score = len(tables) + len(columns)
                    decision_log_sections["Path Identification"].append(
                        f"Path {idx}: Description: {path['description']}, Tables: {', '.join(tables)}, Columns: {', '.join(columns)}, Score: {score}"
                    )

            if tables_and_columns:
                decision_log_sections["Chosen Path Explanation"].append(
                    "Explanation for the chosen path based on table usage and column relevance:"
                )
                for entry in tables_and_columns:
                    table = entry['table']
                    columns = ', '.join(entry['columns'])
                    decision_log_sections["Chosen Path Explanation"].append(
                        f"Table `{table}` used columns `{columns}` due to their relevance and compatibility with data types."
                    )

            if validate_sql_query(query):
                decision_log_sections["Generated SQL Query"].append(
                    f"Here is the SQL query we generated:\n```sql\n{query}\n```"
                )

                natural_language_summary = get_natural_language_summary(query, decision_log_sections["Path Identification"])
                decision_log_sections["Resolution Criteria"].append(
                    f"Summary of the decision-making process: {natural_language_summary}"
                )

                decision_log_sections["Execution Feedback"].append(
                    "SQL query validation was successful. We are now executing the query."
                )

                return {
                    "query": query,
                    "decision_log": build_markdown_decision_log(decision_log_sections),
                    "visualization_recommendation": visualization_recommendation
                }
            else:
                decision_log_sections["Execution Feedback"].append(
                    f"Attempt {attempt + 1}: The SQL query did not pass validation. We're requesting a revision."
                )
                user_message += " Please ensure the query adheres to valid SQL syntax."

        except json.JSONDecodeError:
            decision_log_sections["Execution Feedback"].append(
                f"Attempt {attempt + 1}: We had trouble understanding the server's response. Let's try again."
            )
            decision_log_sections["Execution Feedback"].append(
                "Here's what we received: " + response
            )
            user_message += " The response was not valid JSON. Please provide additional clarity."
            continue

        except Exception as e:
            decision_log_sections["Execution Feedback"].append(
                f"Attempt {attempt + 1}: We ran into an unexpected issue: {e}. Let's try again."
            )
            continue

    # After all attempts failed
    decision_log_sections["Execution Feedback"].append("Final Outcome:")
    decision_log_sections["Execution Feedback"].append(
        "Unfortunately, we couldn't generate a valid SQL query after several attempts."
    )

    return {
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": build_markdown_decision_log(decision_log_sections),
        "visualization_recommendation": None
    }


def build_markdown_decision_log(sections: dict) -> str:
    """
    Constructs a markdown formatted decision log from the provided sections.
    Hides sections with no content.
    """
    markdown_log = ""
    for section, contents in sections.items():
        if contents:
            markdown_log += f"### {section}\n\n"
            for content in contents:
                markdown_log += f"{content}\n\n"
    return markdown_log

def get_natural_language_summary(query: str, paths_summary: list) -> str:
    """
    Generate a natural language summary of the decision process.
    """
    summary_prompt = (
        f"Given the SQL query: '{query}', provide a comprehensive breakdown of the various paths considered for generating this query. "
        f"Explain the decision-making process that led to selecting the specific path used, detailing each step in bullet-point format. "
        f"If multiple paths were encountered, suggest strategies or criteria for resolving these conflicts effectively.\n\n"
        f"{' '.join(paths_summary)}\n"
        f"Provide a concise natural language explanation of the decision-making process, as well as conflict-resolution suggestions."
    )

    response = get_completion_from_messages(SYSTEM_MESSAGE, summary_prompt)

    return response.strip() if response else "Summary Generation Failed."


def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Optional[alt.Chart]:
    """
    Create a chart visualization based on the selected chart type and columns.
    """
    base_chart = alt.Chart(df).configure_title(fontSize=18, fontWeight='bold', font='Roboto')

    try:
        chart_props = {
            "Bar Chart": base_chart.mark_bar(),
            "Line Chart": base_chart.mark_line(),
            "Scatter Plot": base_chart.mark_circle(),
            "Area Chart": base_chart.mark_area(),
            "Histogram": base_chart.mark_bar()
        }

        if chart_type not in chart_props:
            st.warning("Chart type not recognized.")
            return None

        if chart_type == "Histogram":
            chart = chart_props[chart_type].encode(
                alt.X(x_col, bin=alt.Bin(maxbins=30), title=x_col),
                y=alt.Y('count()', title='Count')
            ).properties(
                width='container',
                height=400
            ).interactive()
        else:
            encoding = {
                "x": alt.X(x_col, title=x_col),
                "y": alt.Y(y_col, title=y_col)
            }
            if chart_type in ["Bar Chart", "Line Chart"]:
                encoding["color"] = alt.Color(y_col, legend=None)
            elif chart_type == "Scatter Plot":
                encoding["tooltip"] = [x_col, y_col]

            chart = chart_props[chart_type].encode(**encoding).properties(
                width='container',
                height=400
            ).interactive()

        return chart

    except Exception as e:
        st.error(f"Error generating the chart: {e}")
        logger.error(f"Error generating chart: {e}")
        return None


def display_summary_statistics(df: pd.DataFrame) -> None:
    """
    Display summary statistics of the dataframe.
    """
    if df.empty:
        st.warning("The DataFrame is empty. Unable to display summary statistics.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    tab1, tab2 = st.tabs(["Numeric Summary Statistics", "Categorical Data Insights"])

    if not numeric_cols.empty:
        with tab1:
            numeric_stats = df[numeric_cols].describe().T
            numeric_stats['median'] = df[numeric_cols].median()
            numeric_stats['mode'] = df[numeric_cols].mode().fillna(0).iloc[0]
            numeric_stats['iqr'] = numeric_stats['75%'] - numeric_stats['25%']
            numeric_stats['skew'] = df[numeric_cols].skew()
            numeric_stats['kurt'] = df[numeric_cols].kurt()

            st.markdown("### Numeric Summary Statistics")
            st.dataframe(numeric_stats.style.format("{:.2f}").highlight_max(axis=0, color="lightgreen"))

            for col in numeric_cols:
                st.markdown(f"#### {col}")
                chart = alt.Chart(df).mark_bar().encode(
                    alt.X(col, bin=alt.Bin(maxbins=30), title=f"Distribution of {col}"),
                    y='count()'
                ).properties(
                    width='container',
                    height=200
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

    if not non_numeric_cols.empty:
        with tab2:
            st.markdown("### Categorical Data Insights")
            for col in non_numeric_cols:
                st.markdown(f"**{col} Frequency**")
                freq_table = df[col].value_counts().reset_index()
                freq_table.columns = ['Category', 'Count']
                freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))


def handle_query_response(response: dict, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
    """
    Process the API response and display query results, charts, and decision log with enhanced error handling.
    """
    try:
        query = response.get('query', '')
        error = response.get('error', '')
        decision_log = response.get('decision_log', '')
        visualization_recommendation = response.get('visualization_recommendation', None)

        if error:
            detailed_error = generate_detailed_error_message(error)
            st.error(f"Error generating SQL query: {detailed_error}")
            return

        if not query:
            st.warning("No query generated. Please refine your message.")
            return

        st.success("SQL Query generated successfully!")
        colored_header("SQL Query and Summary", color_name="blue-70", description="")
        st.code(query, language="sql")

        if decision_log:
            with st.expander("Decision Log", expanded=False):
                st.markdown(decision_log)

        sql_results = get_data(query, db_name, db_type, host, user, password)

        if sql_results.empty:
            # Enhanced message explaining why no results were returned
            no_result_reason = "The query executed successfully but did not match any records in the database."
            if 'no valid SQL query generated' in decision_log:
                no_result_reason = "The query was not generated due to insufficient or ambiguous input."
            elif 'SQL query validation failed' in decision_log:
                no_result_reason = "The query failed validation checks and was not executed."
            st.warning(f"The query returned no results because: {no_result_reason}")
            return

        if sql_results.columns.duplicated().any():
            st.error("The query returned a DataFrame with duplicate column names. Please modify your query to avoid this.")
            return

        # Convert date-like columns to datetime
        for col in sql_results.select_dtypes(include=['object']):
            try:
                sql_results[col] = pd.to_datetime(sql_results[col])
            except (ValueError, TypeError):
                pass  # Keep original data type if conversion fails

        colored_header("Query Results and Filter", color_name="blue-70", description="")
        filtered_results = dataframe_explorer(sql_results, case=False)
        st.dataframe(filtered_results, use_container_width=True, height=600)

        colored_header("Summary Statistics and Export Options", color_name="blue-70", description="")
        display_summary_statistics(filtered_results)

        if len(filtered_results.columns) >= 2:
            with st.sidebar.expander("📊 Visualization Options", expanded=True):
                numerical_cols = filtered_results.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = filtered_results.select_dtypes(include=['object', 'category']).columns.tolist()

                suggested_x, suggested_y = None, None

                if numerical_cols:
                    suggested_x = numerical_cols[0]
                    suggested_y = numerical_cols[1] if len(numerical_cols) > 1 else (categorical_cols[0] if categorical_cols else None)
                elif categorical_cols:
                    suggested_x = categorical_cols[0]
                    suggested_y = categorical_cols[1] if len(categorical_cols) > 1 else None

                # Fallback defaults
                if not suggested_x:
                    suggested_x = filtered_results.columns[0] if not filtered_results.columns.empty else 'Column1'
                if not suggested_y:
                    suggested_y = filtered_results.columns[1] if len(filtered_results.columns) > 1 else (filtered_results.columns[0] if not filtered_results.columns.empty else 'Column2')

                # Prepare options with suggestions
                x_options = [f"{col} ⭐" if col == suggested_x else col for col in filtered_results.columns]
                y_options = [f"{col} ⭐" if col == suggested_y else col for col in filtered_results.columns]

                # User selections
                x_col = st.selectbox("Select X-axis Column", options=x_options, index=x_options.index(f"{suggested_x} ⭐") if f"{suggested_x} ⭐" in x_options else 0, key="x_axis")
                y_col = st.selectbox("Select Y-axis Column", options=y_options, index=y_options.index(f"{suggested_y} ⭐") if f"{suggested_y} ⭐" in y_options else 0, key="y_axis")

                x_col_clean = x_col.replace(" ⭐", "")
                y_col_clean = y_col.replace(" ⭐", "")

                chart_type_options = ["None", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram"]
                suggested_chart_type = visualization_recommendation if visualization_recommendation in chart_type_options else ("Bar Chart" if numerical_cols else "None")
                chart_type_display = [f"{chart} ⭐" if chart == suggested_chart_type else chart for chart in chart_type_options]

                try:
                    default_chart_index = chart_type_display.index(f"{suggested_chart_type} ⭐")
                except ValueError:
                    default_chart_index = 0  # Default to "None"

                chart_type = st.selectbox(
                    "Select Chart Type",
                    options=chart_type_display,
                    index=default_chart_index,
                    help=f"Recommended Chart Type: {suggested_chart_type}",
                    key="chart_type"
                )

                chart_type_clean = chart_type.replace(" ⭐", "")

            if chart_type_clean != "None" and x_col_clean and y_col_clean:
                chart = create_chart(filtered_results, chart_type_clean, x_col_clean, y_col_clean)
                if chart:
                    with chart_container(data=filtered_results):
                        st.altair_chart(chart, use_container_width=True)

        export_format = st.selectbox("Select Export Format", options=["CSV", "Excel", "JSON"], key="export_format")
        export_results(filtered_results, export_format)

        # Initialize Query History
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
            st.session_state.query_timestamps = []

        st.session_state.query_history.append(query)
        st.session_state.query_timestamps.append(pd.Timestamp.now())

    except Exception as e:
        detailed_error = generate_detailed_error_message(str(e))
        st.error(f"An unexpected error occurred: {detailed_error}")
        logger.exception(f"Unexpected error: {e}")


def validate_sql_query(query: str) -> bool:
    """
    Validate the SQL query syntax.
    """
    if not isinstance(query, str):
        return False

    disallowed_keywords = r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b'

    if re.search(disallowed_keywords, query, re.IGNORECASE):
        return False

    if not query.strip().lower().startswith(('select', 'with')):
        return False

    if query.count('(') != query.count(')'):
        return False

    return True


def export_results(sql_results: pd.DataFrame, export_format: str) -> None:
    """
    Export the results in the selected format.
    """
    if export_format == "CSV":
        st.download_button(
            label="📥 Download Results as CSV",
            data=sql_results.to_csv(index=False),
            file_name='query_results.csv',
            mime='text/csv'
        )
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sql_results.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_buffer,
            file_name='query_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    elif export_format == "JSON":
        st.download_button(
            label="📥 Download Results as JSON",
            data=sql_results.to_json(orient='records'),
            file_name='query_results.json',
            mime='application/json'
        )
    else:
        st.error("⚠️ Selected export format is not supported.")


def analyze_dataframe_for_visualization(df: pd.DataFrame) -> list:
    """
    Analyze the DataFrame and suggest suitable visualization types based on data characteristics.
    """
    suggestions = set()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.debug(f"Numerical Columns: {numerical_cols}")
    logger.debug(f"Categorical Columns: {categorical_cols}")

    # Single Variable Visualizations
    if len(numerical_cols) == 1:
        suggestions.update(["Histogram", "Box Plot"])
    if len(categorical_cols) == 1:
        suggestions.update(["Bar Chart", "Pie Chart"])

    # Two Variable Visualizations
    if len(numerical_cols) >= 2:
        suggestions.update(["Scatter Plot", "Line Chart"])
    elif len(numerical_cols) == 1 and len(categorical_cols) == 1:
        suggestions.update(["Bar Chart"])

    # Multi-variable or Complex Relationships
    if len(numerical_cols) > 2:
        suggestions.add("Scatter Plot")

    # Time Series Data
    time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        suggestions.add("Line Chart")

    ordered_suggestions = [chart for chart in SUPPORTED_CHART_TYPES.keys() if chart in suggestions]
    logger.debug(f"Ordered Suggestions: {ordered_suggestions}")
    return ordered_suggestions

def generate_detailed_error_message(error_message: str) -> str:
    """
    Use OpenAI to generate a detailed explanation of the error message.
    """
    prompt = f"Provide a detailed and user-friendly explanation for the following error message:\n\n{error_message}"
    detailed_error = get_completion_from_messages(SYSTEM_MESSAGE, prompt)
    return detailed_error.strip() if detailed_error else error_message

# Database Setup
db_type = st.sidebar.selectbox("Select Database Type 🗄️", options=["SQLite", "PostgreSQL"])

if db_type == "SQLite":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite Database 📂", type=["db", "sqlite", "sql"])

    if uploaded_file:
        db_file = save_temp_file(uploaded_file)
        schemas = DB_Config.get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if table_names:
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=options, key="sqlite_tables")
            if "Select All" in selected_tables:
                if len(selected_tables) < len(options):
                    selected_tables = table_names
                else:
                    selected_tables = options
            selected_tables = [table for table in selected_tables if table != "Select All"]
            colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
            for table in selected_tables:
                with st.expander(f"View Schema: {table} 📖", expanded=False):
                    st.json(schemas[table])

            user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message", label="Your Query 💬", label_visibility="hidden")
            if user_message:
                with st.spinner('🧠 Generating SQL query...'):
                    response = cached_generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                handle_query_response(response, db_file, db_type='sqlite')

        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("📥 Please upload a database file to start.")

elif db_type == "PostgreSQL":
    with st.sidebar.expander("🔐 PostgreSQL Connection Details", expanded=True):
        postgres_host = st.text_input("Host 🏠", placeholder="PostgreSQL Host")
        postgres_db = st.text_input("DB Name 🗄️", placeholder="Database Name")
        postgres_user = st.text_input("Username 👤", placeholder="Username")
        postgres_password = st.text_input("Password 🔑", type="password", placeholder="Password")

    if all([postgres_host, postgres_db, postgres_user, postgres_password]):
        schemas = DB_Config.get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=options, key="postgresql_tables")
            if "Select All" in selected_tables:
                if len(selected_tables) < len(options):
                    selected_tables = table_names
                else:
                    selected_tables = options
            selected_tables = [table for table in selected_tables if table != "Select All"]
            colored_header("🔍 Selected Tables:", color_name="blue-70", description="")
            for table in selected_tables:
                with st.expander(f"View Schema: {table} 📖", expanded=False):
                    st.json(schemas[table])

            user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message_pg", label="Your Query 💬", label_visibility="hidden")
            if user_message:
                with st.spinner('🧠 Generating SQL query...'):
                    response = cached_generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)

        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("🔒 Please fill in all PostgreSQL connection details to start.")

# Query history 
with st.sidebar.expander(" Query History", expanded=False):
    if st.session_state.get("query_history"):
        st.write("### 📝 Saved Queries")

        search_query = st.text_input("Search Queries 🔍", key="search_query")
        query_history_df = pd.DataFrame({
            "Query": st.session_state.query_history,
            "Timestamp": pd.to_datetime(st.session_state.query_timestamps)
        })

        if search_query:
            query_history_df = query_history_df[query_history_df['Query'].str.contains(search_query, case=False, na=False)]

        queries_per_page = 5
        total_queries = len(query_history_df)
        num_pages = max((total_queries // queries_per_page) + (total_queries % queries_per_page > 0), 1)
        current_page = st.number_input("Page 📄", min_value=1, max_value=num_pages, value=1)

        start_index = (current_page - 1) * queries_per_page
        end_index = start_index + queries_per_page
        page_queries = query_history_df.iloc[start_index:end_index]

        for i, row in page_queries.iterrows():
            with st.expander(f"🗂️ Query {i + 1}: {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write("**SQL Query:**")
                st.code(row['Query'], language="sql")

                if st.button(f"🔄 Re-run Query {i + 1}", key=f"rerun_query_{i}"):
                    user_message = row['Query']
                    with st.spinner('🔄 Re-running the saved SQL query...'):
                        response = cached_generate_sql_query(user_message, schemas={table: schemas[table] for table in selected_tables})
                        handle_query_response(
                            response,
                            db_file if db_type == "SQLite" else postgres_db,
                            db_type,
                            host=postgres_host if db_type == "PostgreSQL" else None,
                            user=postgres_user if db_type == "PostgreSQL" else None,
                            password=postgres_password if db_type == "PostgreSQL" else None
                        )

        st.write(f"Page {current_page} of {num_pages}")

    else:
        st.info("📭 No query history available.")