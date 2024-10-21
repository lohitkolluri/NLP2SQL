import sqlite3
import psycopg2
from sqlite3 import Error
import pandas as pd

def create_connection(db_name, host=None, user=None, password=None):
    """Create or connect to a database."""
    conn = None
    if host:  # PostgreSQL connection
        try:
            conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host
            )
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
    else:  # SQLite connection
        try:
            conn = sqlite3.connect(db_name)
        except Error as e:
            print(e)
    return conn

def query_database(query, db_name, db_type, host=None, user=None, password=None):
    """Run an SQL query and return results in a DataFrame."""
    conn = create_connection(db_name, host, user, password)
    if conn is None:
        return pd.DataFrame()  # Return empty DataFrame on connection failure
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    return df

def get_all_schemas(db_name, db_type, host=None, user=None, password=None):
    """Retrieve schema representation of all tables in the database."""
    conn = create_connection(db_name, host, user, password)
    if conn is None:
        return {}  # Return empty dict on connection failure

    cursor = conn.cursor()
    
    # Get all table names
    if db_type == 'sqlite':
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    elif db_type == 'postgresql':
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    
    tables = cursor.fetchall()
    
    schemas = {}
    for table in tables:
        table_name = table[0]
        if db_type == 'sqlite':
            cursor.execute(f"PRAGMA table_info({table_name});")
        else:
            cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
        
        columns = cursor.fetchall()
        
        schema = {}
        for col in columns:
            if db_type == 'sqlite':
                schema[col[1]] = col[2]  # col[1] is the column name, col[2] is the data type
            elif db_type == 'postgresql':
                schema[col[0]] = col[1]  # col[0] is the column name, col[1] is the data type
        
        schemas[table_name] = schema
    
    conn.close()
    return schemas
