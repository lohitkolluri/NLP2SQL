# NLP2SQL

NLP2SQL is a project that aims to generate SQL queries from natural language text. It involves converting text input into a structured format to create a semantically correct SQL query for database execution.

## 🧐 Features

Here are some of the project's best features:

- **Natural Language to SQL**: Convert natural language text to SQL queries.
- **Dynamic Database Handling**: Adapt to different database schemas dynamically.
- **Data Display and Visualization**: Present data in an intuitive and interactive manner.
- **Report Generation**: Generate comprehensive reports based on SQL queries.
- **Performance Optimization**: Optimized for efficiency and speed.

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

   Create a `.env` file in the root directory and add the necessary environment variables such as your Azure OpenAI endpoint and API key.

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

- **Streamlit**: For building the interactive web application.
- **Altair**: For data visualization.
- **Python**: The core programming language.
- **SQLite**: For the database.
- **Pandas**: For data manipulation.
- **Azure OpenAI**: For generating SQL queries from natural language input.
