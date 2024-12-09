# NLP2SQL 🤖

<div align="center">

[![Made with Azure OpenAI](https://img.shields.io/badge/Made%20with-Azure%20OpenAI-0078D4?style=for-the-badge&logo=microsoftazure)](https://azure.microsoft.com/products/cognitive-services/openai-service/)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Database](https://img.shields.io/badge/Database-PostgreSQL%20%7C%20SQLite-336791?style=for-the-badge&logo=postgresql)](https://www.postgresql.org/)

> 🎯 Transform natural language into powerful SQL queries with ease!

</div>

<p align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Magnifying%20Glass%20Tilted%20Right.png" alt="Magnifying Glass" width="25" height="25" /> Turn your words into SQL magic
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Chart%20Increasing.png" alt="Chart" width="25" height="25" /> Visualize your data dynamically
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bookmark%20Tabs.png" alt="Tabs" width="25" height="25" /> Multi-database support
</p>

## ✨ Features

<details>
<summary>🎯 Core Features</summary>

- **Natural Language to SQL** 🗣️ → 📝
  - Convert text queries into SQL commands using Azure OpenAI
  - Intelligent query interpretation and generation
  
- **Multi-Database Support** 🗄️
  - SQLite compatibility
  - PostgreSQL integration
  - Dynamic database switching
  
- **Interactive Data Explorer** 🔍
  - Real-time data filtering
  - Interactive query results
  - Dynamic table views

</details>

<details>
<summary>📊 Visualization & Analytics</summary>

- **Dynamic Visualizations** 📈
  - Multiple chart types
  - Interactive configurations
  - Smart visualization suggestions ⭐️
  
- **Summary Statistics** 📋
  - Numeric data insights
  - Categorical analysis
  - Trend identification

</details>

<details>
<summary>🛡️ Security & Management</summary>

- **Safe SQL Execution** 🔒
  - Query validation
  - SQL injection prevention
  - Error handling
  
- **Query History** 📚
  - Searchable log
  - Re-runnable queries
  - Export capabilities

</details>

## 🚀 Getting Started

```mermaid
graph LR
    A[User Input] --> B[NLP Engine]
    B --> C[SQL Generator]
    C --> D[Database]
    D --> E[Results]
    E --> F[Visualization]
```


### Installation

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/lohitkolluri/NLP2SQL.git
cd NLP2SQL
```

2️⃣ **Set Up Environment**
```bash
# Create .env file
cat << EOF > .env
OPENAI_API_KEY="Your Azure OpenAI API Key"
OPENAI_ENDPOINT="https://name_of_openai_resource.openai.azure.com/"
OPENAI_API_VERSION="API Version"
MODEL_NAME="Name of Your Model from Azure OpenAI"
EOF
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Launch the App**
```bash
streamlit run NLP2SQL.py
```

## 🎨 Built With

<div align="center">

| Technology | Purpose |
|:-----------:|:--------:|
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit) | Web Interface |
| ![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-0078D4?style=flat-square&logo=microsoftazure) | NLP Processing |
| ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql) | Database |
| ![Altair](https://img.shields.io/badge/Altair-005571?style=flat-square) | Visualizations |

</div>

## 🌟 Key Features

```mermaid
mindmap
  root((NLP2SQL))
    Query Processing
      Natural Language Input
      SQL Generation
      Validation
    Visualization
      Charts
      Tables
      Statistics
    Database
      PostgreSQL
      SQLite
      Schema Explorer
    Security
      Query Validation
      Error Handling
      Safe Execution
```

## 💡 How It Works

1. **Query Input** ➡️ User enters natural language query
2. **Processing** ➡️ Azure OpenAI translates to SQL
3. **Execution** ➡️ Query runs against selected database
4. **Visualization** ➡️ Results displayed with appropriate charts
5. **Export** ➡️ Download results in preferred format

<div align="center">

### 🌟 Start Exploring Your Data Today! 
---
</div>
