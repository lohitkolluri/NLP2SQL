import io
import json
import re
import sql_db

import pandas as pd
import altair as alt
import streamlit as st
import streamlit_nested_layout
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
import numpy as np
import logging

from sql_db import *
from prompts.prompts import SYSTEM_MESSAGE
from streamlit_extras.chart_container import chart_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
from azure_openai import get_completion_from_messages

# Set page configuration with a custom layout
st.set_page_config(
    page_icon="🗃️",
    page_title="Chat with Your DB",
    layout="wide"
)

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


# Cache data fetching function
@st.cache_data
def get_data(query: str, db_name: str, db_type: str, host: str = None, user: str = None, password: str = None) -> pd.DataFrame:
    """
    Function to fetch results from the database based on the provided SQL query.
    Arguments:
    query : str : SQL query
    db_name : str : name of the database
    db_type : str : type of the database
    host : str : optional host
    user : str : optional user
    password : str : optional password
    Returns:
    df : pd.DataFrame : Dataframe with the query results
    """
    return sql_db.query_database(query, db_name, db_type, host, user, password)

# Save uploaded database file temporarily
def save_temp_file(uploaded_file) -> str:
    """
    Function to save the uploaded database file temporarily.
    Arguments:
    uploaded_file : UploadedFile : streamlit uploaded file
    """
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

# Cache the generate_sql_query function
@st.cache_data
def cached_generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 3) -> dict:
    return generate_sql_query(user_message, schemas, max_attempts)

# Generate SQL query from user input
def generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 3) -> dict:
    formatted_system_message = SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))
    decision_log = []
    paths_summary = []
    paths_data = []
    visualization_recommendation = None

    for attempt in range(max_attempts):
        response = get_completion_from_messages(formatted_system_message, user_message)

        try:
            json_response = json.loads(response)
            query = json_response.get('query', None)
            error = json_response.get('error', None)
            paths_considered = json_response.get('paths_considered', [])
            final_choice = json_response.get('final_choice', '')
            tables_and_columns = json_response.get('tables_and_columns', [])
            visualization_recommendation = json_response.get('visualization_recommendation', None)

            decision_log.append(f"Attempt {attempt + 1}:")

            if error:
                decision_log.append(f"Error encountered: {error}. Action: Retrying with adjusted parameters.")
                continue

            if query is None:
                decision_log.append("Warning: No valid SQL query was generated. Action: Retrying with additional context.")
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
                    decision_log.append(f"{idx}. Description: {path['description']} | Tables: {tables_str} | Columns: {columns_str}")
                    paths_summary.append(f"Path {idx}: {path['description']} using tables `{tables_str}` and columns `{columns_str}`.")

            if tables_and_columns:
                decision_log.append("Tables and Columns Utilized:")
                for entry in tables_and_columns:
                    table = entry['table']
                    columns = ', '.join(entry['columns'])
                    decision_log.append(f"- Table: `{table}` | Columns: `{columns}`")

            if validate_sql_query(query):
                decision_log.append("Validation: The SQL query passed all validation checks.")
                
                natural_language_summary = get_natural_language_summary(query, paths_summary)
                decision_log.append("Summary of Decision-Making Process:")
                decision_log.append(natural_language_summary)

                if paths_data:
                    best_path = min(paths_data, key=lambda x: x['score'])
                    final_choice = best_path['description']
                    decision_log.append(f"Final Decision: Selected the best path based on scoring: `{final_choice}`.")

                return {
                    "query": query,
                    "decision_log": "\n".join(decision_log),
                    "visualization_recommendation": visualization_recommendation
                }
            else:
                decision_log.append("Validation: The SQL query failed validation checks. Action: Requesting a revised query with correct SQL syntax.")
                user_message += " Please ensure that the query adheres to valid SQL syntax."

        except json.JSONDecodeError:
            decision_log.append(f"Attempt {attempt + 1}:")
            decision_log.append("Error: Failed to decode JSON response. Action: Retrying with clearer instructions.")
            decision_log.append(f"Raw Response: `{response}`")
            user_message += " The response was not valid JSON. Please provide additional clarity."
            continue

        except Exception as e:
            decision_log.append(f"Attempt {attempt + 1}:")
            decision_log.append(f"Unexpected Error: `{e}`. Action: Retrying due to unexpected error.")
            decision_log.append(f"Error Details: `{e}`")
            continue

    # After all attempts failed
    decision_log.append("Final Outcome:")
    decision_log.append("Status: Failed to generate a valid SQL query after multiple attempts.")
    decision_log.append("Actions Taken:\n- Attempted to generate SQL query multiple times with adjusted parameters.\n- Reviewed and validated generated queries.\n- Encountered persistent errors preventing successful query generation.")
    
    return {
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": "\n".join(decision_log),
        "visualization_recommendation": None
    }

# Generate natural language summary of decision process
def get_natural_language_summary(query: str, paths_summary: list) -> str:
    """
    Function to generate a natural language summary of the selected query paths.
    Arguments:
    query : str : the generated SQL query
    paths_summary : list : list of paths considered
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

# Create chart visualization
def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> alt.Chart:
    """
    Function to create a chart visualization.
    Arguments:
        df : pd.DataFrame : dataframe to be used
        chart_type : str : type of the chart
        x_col : str : column for X axis
        y_col : str : column for Y axis
    Returns:
        chart : alt.Chart : created chart
    """
    base_chart = alt.Chart(df).properties(width=600, height=400).configure_title(fontSize=18, fontWeight='bold', font='Roboto')

    try:
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
            st.warning("Chart type not recognized.")
            return None

        return chart
    except Exception as e:
        st.error(f"An error occurred while generating the chart: {e}")
        logging.error(f"Error generating chart: {e}")
        return None

# Display summary statistics of a dataframe
def display_summary_statistics(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("The DataFrame is empty, unable to display summary statistics.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    if numeric_cols.any():
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats['median'] = df[numeric_cols].median()
        numeric_stats['mode'] = df[numeric_cols].mode().iloc[0]
        numeric_stats['iqr'] = numeric_stats['75%'] - numeric_stats['25%']
        numeric_stats['skew'] = df[numeric_cols].skew()
        numeric_stats['kurt'] = df[numeric_cols].kurt()

        # Display numeric summary statistics with histograms
        st.markdown("### Numeric Summary Statistics")
        for col in numeric_cols:
            st.markdown(f"#### {col}")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(numeric_stats.loc[[col]].style.format("{:.2f}").highlight_max(axis=0, color="lightgreen"))

            with col2:
                st.altair_chart(alt.Chart(df).mark_bar().encode(
                    alt.X(col, bin=alt.Bin(maxbins=30), title=f"Distribution of {col}"),
                    y='count()'
                ).properties(width=350, height=200), use_container_width=True)

    if non_numeric_cols.any():
        st.markdown("### Categorical Data Insights")
        for col in non_numeric_cols:
            st.markdown(f"**{col} Frequency**")
            freq_table = df[col].value_counts().reset_index()
            freq_table.columns = ['Category', 'Count']
            freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
            st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))

# Handle query response and display results
def handle_query_response(response: dict, db_name: str, db_type: str, host: str = None, user: str = None, password: str = None) -> None:
    """
    Function to process the API response and display query results, charts, and decision log.
    Arguments:
        response : dict : response from the API containing 'query', 'decision_log', and 'visualization_recommendation'
        db_name : str : name of the database
        db_type : str : type of the database
        host : str : optional host
        user : str : optional user
        password : str : optional password
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
        colored_header("SQL Query and Summary", color_name="blue-70")
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

        for col in sql_results.select_dtypes(include=['object']):
            try:
                sql_results[col] = pd.to_datetime(sql_results[col])
            except ValueError:
                pass  # If conversion fails, keep the original data type

        colored_header("Query Results and Filter", color_name="blue-70")
        filtered_results = dataframe_explorer(sql_results, case=False)
        st.dataframe(filtered_results, use_container_width=True, height=600, width=1000)

        colored_header("Summary Statistics and Export Options", color_name="blue-70")
        display_summary_statistics(filtered_results)

        if len(filtered_results.columns) >= 2:
            st.sidebar.markdown("### Visualization Options")

            # Categorize columns
            numerical_cols = filtered_results.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = filtered_results.select_dtypes(include=['object', 'category']).columns.tolist()

            # Determine suggested X and Y columns
            suggested_x = None
            suggested_y = None

            if numerical_cols:
                suggested_x = numerical_cols[0]
                suggested_y = numerical_cols[1] if len(numerical_cols) > 1 else (categorical_cols[0] if categorical_cols else None)
            elif categorical_cols:
                suggested_x = categorical_cols[0]
                suggested_y = categorical_cols[1] if len(categorical_cols) > 1 else None

            # Fallback defaults if no suitable columns are found
            if not suggested_x:
                suggested_x = filtered_results.columns[0] if filtered_results.columns else 'Column1'
            if not suggested_y:
                suggested_y = filtered_results.columns[1] if len(filtered_results.columns) > 1 else (filtered_results.columns[0] if filtered_results.columns else 'Column2')

            # Prepare X-axis options with suggestion
            x_options = [f"{col} ⭐" if col == suggested_x else col for col in filtered_results.columns]

            # Prepare Y-axis options with suggestion
            y_options = [f"{col} ⭐" if col == suggested_y else col for col in filtered_results.columns]

            # User selects X and Y columns
            x_col = st.sidebar.selectbox("Select X-axis Column", options=x_options, index=x_options.index(f"{suggested_x} ⭐") if f"{suggested_x} ⭐" in x_options else 0)
            y_col = st.sidebar.selectbox("Select Y-axis Column", options=y_options, index=y_options.index(f"{suggested_y} ⭐") if f"{suggested_y} ⭐" in y_options else 0)

            # Clean selections by removing the star
            x_col_clean = x_col.replace(" ⭐", "")
            y_col_clean = y_col.replace(" ⭐", "")

            # Determine suggested Chart Type
            chart_type_options = ["None", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram"]
            suggested_chart_type = visualization_recommendation if visualization_recommendation in chart_type_options else ("Bar Chart" if numerical_cols else "None")

            # Append star to the suggested chart type
            chart_type_display = [f"{chart} ⭐" if chart == suggested_chart_type else chart for chart in chart_type_options]

            # Find the index of the suggested chart type
            try:
                default_chart_index = chart_type_display.index(f"{suggested_chart_type} ⭐")
            except ValueError:
                default_chart_index = 0  # Default to "None" if not found

            # User selects Chart Type
            chart_type = st.sidebar.selectbox(
                "Select Chart Type",
                options=chart_type_display,
                index=default_chart_index,
                help=f"Recommended Chart Type: {suggested_chart_type}"
            )

            # Clean chart type selection
            chart_type_clean = chart_type.replace(" ⭐", "")

            if chart_type_clean != "None" and x_col_clean and y_col_clean:
                chart = create_chart(filtered_results, chart_type_clean, x_col_clean, y_col_clean)
                if chart:
                    with chart_container(data=filtered_results, export_formats=["CSV", "Parquet"]):
                        st.altair_chart(chart)

        export_format = st.selectbox("Select Export Format", options=["CSV", "Excel", "JSON"])
        export_results(filtered_results, export_format)

        if "query_history" not in st.session_state:
            st.session_state.query_history = []
            st.session_state.query_timestamps = []

        st.session_state.query_history.append(query)
        st.session_state.query_timestamps.append(pd.Timestamp.now())

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Validate SQL query syntax
def validate_sql_query(query: str) -> bool:
    """
    Function to validate the SQL query.
    Arguments:
    query : str : SQL query
    Returns:
    bool : whether the SQL query is valid
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

# Export results in selected format
def export_results(sql_results: pd.DataFrame, export_format: str) -> None:
    """
    Function to export the results in the selected format.
    Arguments:
    sql_results : pd.DataFrame : dataframe with the results
    export_format : str : format for the export
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

# Database selection and connection settings
db_type = st.sidebar.selectbox("Select Database Type 🗄️", options=["SQLite", "PostgreSQL"])

if db_type == "SQLite":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite Database 📂", type=["db", "sqlite", "sql"])

    if uploaded_file is not None:
        db_file = save_temp_file(uploaded_file)
        schemas = get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if table_names:
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=table_names, format_func=lambda x: f"{x} ")
            if selected_tables:
                colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
                for table in selected_tables:
                    with st.expander(f"View Schema: {table} 📖", expanded=False):
                        st.json(schemas[table])

                user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message", label="Your Query 💬", label_visibility="visible")
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
        postgres_host = st.text_input(placeholder="PostgreSQL Host", label="Host 🏠", label_visibility="visible")
        postgres_db = st.text_input(placeholder="Database Name", label="DB Name 🗄️", label_visibility="visible")
        postgres_user = st.text_input(placeholder="Username", label="Username 👤", label_visibility="visible")
        postgres_password = st.text_input(placeholder="Password", type="password", label="Password 🔑", label_visibility="visible")

    if postgres_host and postgres_db and postgres_user and postgres_password:
        schemas = get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=table_names, format_func=lambda x: f"{x} 🗂️")
            if selected_tables:
                colored_header("🔍 Selected Tables:", color_name="blue-70", description="")
                st.markdown(f"<div class='main-header'>Selected Tables: {', '.join(selected_tables)} 🗄️</div>", unsafe_allow_html=True)
                for table in selected_tables:
                    with st.expander(f"View Schema: {table} 📖", expanded=False):
                        st.json(schemas[table])

                user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message_pg", label="Your Query 💬", label_visibility="visible")
                if user_message:
                    with st.spinner('🧠 Generating SQL query...'):
                        response = cached_generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
                    handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)

            else:
                st.info("📭 No tables found in the database.")
        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("🔒 Please fill in all PostgreSQL connection details to start.")

# Query history with re-run and delete options
with st.sidebar.expander(" Query History", expanded=False):
    if "query_history" in st.session_state and st.session_state.query_history:
        st.write("### 📝 Saved Queries")

        search_query = st.text_input(placeholder="Search Queries 🔍", label="Search Queries", label_visibility="visible", key="search_query")
        query_history_df = pd.DataFrame({
            "Query": st.session_state.query_history,
            "Timestamp": pd.to_datetime(st.session_state.query_timestamps)
        })

        if search_query:
            query_history_df = query_history_df[query_history_df['Query'].str.contains(search_query, case=False)]

        queries_per_page = 5
        total_queries = len(query_history_df)
        num_pages = (total_queries // queries_per_page) + (total_queries % queries_per_page > 0)
        current_page = st.number_input("Page 📄", min_value=1, max_value=num_pages, value=1)

        start_index = (current_page - 1) * queries_per_page
        end_index = start_index + queries_per_page
        page_queries = query_history_df.iloc[start_index:end_index]

        for i, (past_query, timestamp) in page_queries.iterrows():
            with st.expander(f"🗂️ Query {i + 1}: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write("**SQL Query:**")
                st.code(past_query, language="sql")

                if st.button(f"🔄 Re-run Query {i + 1}", key=f"rerun_query_{i}"):
                    user_message = past_query
                    with st.spinner('🔄 Re-running the saved SQL query...'):
                        response = cached_generate_sql_query(user_message, {table: schemas[table] for table in selected_tables})
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

def analyze_dataframe_for_visualization(df: pd.DataFrame) -> list:
    """
    Analyze the DataFrame and suggest suitable visualization types based on data characteristics.
    
    Arguments:
        df : pd.DataFrame : The DataFrame to analyze.
    
    Returns:
        suggestions : list : A list of recommended visualization types.
    """
    suggestions = set()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logging.debug(f"Numerical Columns: {numerical_cols}")
    logging.debug(f"Categorical Columns: {categorical_cols}")
    
    # Single Variable Visualizations
    if len(numerical_cols) == 1:
        suggestions.update(["Histogram", "Box Plot"])
    elif len(categorical_cols) == 1:
        suggestions.update(["Bar Chart", "Pie Chart"])
    
    # Two Variable Visualizations
    if len(numerical_cols) >= 2:
        suggestions.update(["Scatter Plot", "Line Chart"])
    elif len(numerical_cols) == 1 and len(categorical_cols) == 1:
        suggestions.update(["Bar Chart"])
    
    # Multi-variable or Complex Relationships
    if len(numerical_cols) > 2:
        suggestions.update(["Scatter Plot"])
    
    # Additional Checks (Optional)
    # Example: Check for time series data based on column names or data patterns
    time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        suggestions.add("Line Chart")
    
    ordered_suggestions = [chart for chart in SUPPORTED_CHART_TYPES.keys() if chart in suggestions]
    logging.debug(f"Ordered Suggestions: {ordered_suggestions}")
    return ordered_suggestions
