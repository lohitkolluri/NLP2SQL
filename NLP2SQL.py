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

import sql_db
from prompts.prompts import SYSTEM_MESSAGE
from streamlit_extras.chart_container import chart_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
from azure_openai import get_completion_from_messages

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

# Set page configuration with a custom layout
st.set_page_config(
    page_icon="🗃️",
    page_title="Chat with Your DB",
    layout="wide"
)

# Load environment variables once
load_dotenv()

# Custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Centered main header */
        .main-header {
            text-align: center;
            color: #1E90FF;
        }

        /* Styled buttons */
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 15em;
            font-size:16px;
            font-weight: bold;
        }

        /* Styled expander headers */
        .st-expanderHeader {
            font-size: 18px;
            font-weight: bold;
            color: #1E90FF;
        }

        /* Sidebar headers */
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }

        /* Improve slider appearance */
        .st-slider > div > div > div > div > div {
            background: #1E90FF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()


@st.cache_resource
def load_system_message(schemas: dict) -> str:
    return SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))


@st.cache_data
def get_data(query: str, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch results from the database based on the provided SQL query.
    """
    return sql_db.query_database(query, db_name, db_type, host, user, password)


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


def generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 3) -> dict:
    """
    Generate SQL query from user input with retry mechanism.
    """
    formatted_system_message = load_system_message(schemas)
    decision_log = []
    paths_summary = []
    paths_data = []
    visualization_recommendation = None

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

            decision_log.append(f"Attempt {attempt + 1}:")

            if error:
                decision_log.append(f"Error encountered: {error}. Retrying...")
                continue

            if not query:
                decision_log.append("No valid SQL query generated. Retrying...")
                continue

            if paths_considered:
                decision_log.append("Paths Considered:")
                for idx, path in enumerate(paths_considered, start=1):
                    tables = path['tables']
                    columns = [col for sublist in path['columns'] for col in sublist]
                    paths_data.append({
                        'description': path['description'],
                        'tables': tables,
                        'columns': columns,
                        'score': len(tables) + len(columns)
                    })
                    tables_str = ', '.join(tables)
                    columns_str = ', '.join(columns)
                    decision_log.append(f"{idx}. {path['description']} | Tables: {tables_str} | Columns: {columns_str}")
                    paths_summary.append(f"Path {idx}: {path['description']} using `{tables_str}` and `{columns_str}`.")

            if tables_and_columns:
                decision_log.append("Tables and Columns Utilized:")
                for entry in tables_and_columns:
                    table = entry['table']
                    columns = ', '.join(entry['columns'])
                    decision_log.append(f"- `{table}`: {columns}")

            if validate_sql_query(query):
                decision_log.append("SQL query validation passed.")
                
                natural_language_summary = get_natural_language_summary(query, paths_summary)
                decision_log.append("Decision Process Summary:")
                decision_log.append(natural_language_summary)

                if paths_data:
                    best_path = min(paths_data, key=lambda x: x['score'])
                    final_choice = best_path['description']
                    decision_log.append(f"Final Decision: Selected `{final_choice}` based on scoring.")

                return {
                    "query": query,
                    "decision_log": "\n".join(decision_log),
                    "visualization_recommendation": visualization_recommendation
                }
            else:
                decision_log.append("SQL query validation failed. Requesting revision.")
                user_message += " Please ensure the query adheres to valid SQL syntax."

        except json.JSONDecodeError:
            decision_log.append("Failed to decode JSON response. Retrying...")
            decision_log.append(f"Raw Response: `{response}`")
            user_message += " The response was not valid JSON. Provide additional clarity."
            continue

        except Exception as e:
            decision_log.append(f"Unexpected error: `{e}`. Retrying...")
            decision_log.append(f"Error Details: `{e}`")
            continue

    # After all attempts failed
    decision_log.append("Final Outcome:")
    decision_log.append("Failed to generate a valid SQL query after multiple attempts.")
    
    return {
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": "\n".join(decision_log),
        "visualization_recommendation": None
    }


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
            )
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
            )

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
                st.altair_chart(
                    alt.Chart(df).mark_bar().encode(
                        alt.X(col, bin=alt.Bin(maxbins=30), title=f"Distribution of {col}"),
                        y='count()'
                    ).properties(
                        width='container',
                        height=200
                    ),
                    use_container_width=True
                )

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
    Process the API response and display query results, charts, and decision log.
    """
    try:
        query = response.get('query', '')
        error = response.get('error', '')
        decision_log = response.get('decision_log', '')
        visualization_recommendation = response.get('visualization_recommendation', None)

        if error:
            st.error(f"Error generating SQL query: {error}")
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
            st.warning("The query returned no results.")
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

        # Initialize query history if not present
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
            st.session_state.query_timestamps = []

        st.session_state.query_history.append(query)
        st.session_state.query_timestamps.append(pd.Timestamp.now())

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
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

# Database selection and connection settings
db_type = st.sidebar.selectbox("Select Database Type 🗄️", options=["SQLite", "PostgreSQL"])

if db_type == "SQLite":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite Database 📂", type=["db", "sqlite", "sql"])

    if uploaded_file:
        db_file = save_temp_file(uploaded_file)
        schemas = sql_db.get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if table_names:
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=table_names)
            if selected_tables:
                colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
                for table in selected_tables:
                    with st.expander(f"View Schema: {table} 📖", expanded=False):
                        st.json(schemas[table])

                user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message", label="Your Query 💬")
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
        schemas = sql_db.get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=table_names)
            if selected_tables:
                colored_header("🔍 Selected Tables:", color_name="blue-70", description="")
                for table in selected_tables:
                    with st.expander(f"View Schema: {table} 📖", expanded=False):
                        st.json(schemas[table])

                user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message_pg", label="Your Query 💬")
                if user_message:
                    with st.spinner('🧠 Generating SQL query...'):
                        response = cached_generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                    handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)

        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("🔒 Please fill in all PostgreSQL connection details to start.")

# Query history with re-run and delete options
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
