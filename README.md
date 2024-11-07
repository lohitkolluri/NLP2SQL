# NLP2SQL

NLP2SQL is a project that aims to generate SQL queries from natural language text. The system converts natural language inputs into semantically correct SQL queries, enabling dynamic database handling, data visualization, and interactive querying.

## 🧐 Features

Here are some of the project's key features:

- **Natural Language to SQL**: Converts text-based user queries into SQL commands.
- **Multi-Database Support**: Connects and operates with SQLite and PostgreSQL databases.
- **Interactive Data Explorer**: Allows filtering and analysis of query results within an interactive data table.
- **Dynamic Schema Representation**: Retrieves and displays database schemas, with options to select tables for queries.
- **Custom Visualizations**: Offers a variety of chart types (Bar, Line, Scatter, Area, Histogram) for visualizing query results.
- **Summary Statistics**: Displays numeric summary statistics (mean, median, mode, IQR, skewness, kurtosis) and categorical data insights for query results.
- **Decision Log and Query Path Summary**: Provides a decision log detailing paths considered and chosen during query generation, explaining ambiguities and decisions made.
- **Export Options**: Supports exporting query results in CSV, Excel, and JSON formats.
- **Query History**: Saves and revisits past queries for easy future access.
- **Safe SQL Execution**: Validates SQL queries to prevent harmful commands like `DROP`, `DELETE`, or `ALTER`.
- **PostgreSQL Connection Configuration**: Allows configuration of host, database, user, and password for PostgreSQL databases.

## 🛠️ Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/lohitkolluri/NLP2SQL.git
   cd NLP2SQL
   ```

2. **Create and Activate a Virtual Environment:**

   - On macOS/Linux:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Create a `.env` file in the root directory and add the necessary environment variables for Azure OpenAI API:

   ```env
   OPENAI_ENDPOINT=https://your-azure-openai-endpoint
   OPENAI_API_KEY=your-azure-openai-api-key
   ```

5. **Running the App:**

   ```bash
   streamlit run NLP2SQL.py
   ```

6. **Access the App:**

   Open your web browser and go to:

   ```
   http://localhost:8501
   ```

## 💻 Built With

Technologies used in the project:

- **Streamlit**: Powers the interactive web application.
- **Altair**: Renders dynamic and customizable data visualizations.
- **Pandas**: Provides robust data manipulation tools.
- **SQLite & PostgreSQL**: Enables multi-database connectivity and support.
- **Azure OpenAI**: Translates natural language queries into SQL commands.
- **Streamlit Extras**: Enhances UI elements such as data exploration tables and chart containers.
- **GPT-3.5 Turbo**: Provides intelligent language processing for query generation.

## 🚀 Additional Features

- **Custom CSS Styling**: Enhances interface aesthetics and user experience with tailored CSS.
- **Schema Explorer**: Displays the database schema to guide query formulation.
- **Database File Upload**: Allows users to upload SQLite database files directly for querying.
- **Collapsible Query History**: Saves query history in a collapsible section for easy access and re-running of queries.
- **Summary Statistics**: Provides detailed numeric and categorical summaries for query results.
- **Dynamic Query Generation**: Manages ambiguous queries effectively, with decision logs explaining interpretation paths.

## 📝 How It Works

The application processes natural language queries, converting them into SQL commands to interact with connected databases. Key functionalities include:

- **Loading Database Schemas**: Automatically retrieves and displays schemas, allowing users to select tables for querying.
- **Query Generation and Validation**: Generates SQL queries with safety checks to prevent destructive commands.
- **Data Visualization**: Creates rich visual representations of data with a range of chart options, powered by Altair.
- **Exporting Results**: Allows users to download query results in multiple formats (CSV, Excel, JSON).
- **Interactive Elements**: Utilizes Streamlit components for a responsive and intuitive user experience.

This application is designed to simplify data interaction, enabling users to explore and visualize their databases effortlessly.

## 🖼️ Diagram

<p align="center">
    <img src="NLP2SQL.png" alt="NLP2SQL Diagram" width="600px">
</p>
