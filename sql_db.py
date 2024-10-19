import sqlite3
from sqlite3 import Error
import pandas as pd

def create_connection(db_file):
    """Create or connect to an SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def query_database(query, db_file):
    """Run an SQL query and return results in a DataFrame."""
    conn = create_connection(db_file)
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

def get_schema_representation(db_file):
    """Get the database schema in a JSON-like format."""
    conn = create_connection(db_file)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    db_schema = {}
    
    for table in tables:
        table_name = table[0]
        
        # Get column details for each table
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        column_details = {column[1]: column[2] for column in columns}
        db_schema[table_name] = column_details
    
    conn.close()
    return db_schema
