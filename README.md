<h1 align="center" id="title">NLP2SQL</h1>

<p id="description">NLP2SQL is an NLP task that aims to generate SQL queries from natural language text. It involves converting text input into a structured format to create a semantically correct SQL query for database execution.</p>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Natural Language to SQL
*   Dynamic Database Handling
*   Data Display and Visualization
*   Report Generation
*   Performance Optimization

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the Repository:</p>

```
git clone https://github.com/lohitkolluri/NLP2SQL.git
```

```
cd NLP2SQL
```

<p>3. Create and Activate a Virtual Environment:</p>

```
# On macOS/Linux 
```

```
python3 -m venv venv 
```

```
source venv/bin/activate
```

```
# On Windows 
```

```
python -m venv venv 
```

```
.\venv\Scripts\activate
```

<p>10. Install Dependencies:</p>

```
pip install -r requirements.txt
```

<p>11. Set Up Environment Variables:</p>

```
Create a .env file in the root directory and add the necessary environment variables such as your Azure OpenAI endpoint and API key.
```

```
OPENAI_ENDPOINT=https://your-azure-openai-endpoint
```

```
OPENAI_API_KEY=your-azure-openai-api-key
```

<p>14. Running the App</p>

```
streamlit run main_app_streamlit.py
```

<p>15. Access the App:</p>

```
http://localhost:8501
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Streamlit
*   Plotly
*   Python
*   SQLite
*   Pandas
*   Azure OpenAI
