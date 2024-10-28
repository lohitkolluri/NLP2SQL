import sqlite3
import psycopg2
from sqlite3 import Error
import pandas as pd

def create_connection(db_name, host=None, user=None, password=None):
    """Create or connect to a database."""
    conn = None
    try:
        if host:  # PostgreSQL connection
            conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host
            )
        else:  # SQLite connection
            conn = sqlite3.connect(db_name)
    except Exception as e:
        print(f"Error connecting to the database: {e}")
    return conn

def query_database(query, db_name, db_type, host=None, user=None, password=None):
    """Run an SQL query and return results in a DataFrame."""
    conn = create_connection(db_name, host, user, password)
    if conn is None:
        return pd.DataFrame()

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
        return {}

    cursor = conn.cursor()
    schemas = {}
    
    try:
        # Get all table names
        if db_type == 'sqlite':
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        elif db_type == 'postgresql':
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            if db_type == 'sqlite':
                cursor.execute(f"PRAGMA table_info({table_name});")
            else:
                cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
            
            columns = cursor.fetchall()
            schema = {col[0]: col[1] for col in columns}
            schemas[table_name] = schema
    except Exception as e:
        print(f"Error retrieving schemas: {e}")
    finally:
        conn.close()

    return schemas
