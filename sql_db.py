import sqlite3
from sqlite3 import Error
import random
from datetime import date, timedelta
import pandas as pd

def create_connection(db_file):
    """Create or connect to an SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """Create a table with the specified SQL command."""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def insert_data(conn, table_name, data_dict):
    """Insert new data into a table."""
    columns = ', '.join(data_dict.keys())
    placeholders = ', '.join('?' * len(data_dict))
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    cur = conn.cursor()
    cur.execute(sql, list(data_dict.values()))
    conn.commit()
    return cur.lastrowid

def query_database(query, db_file):
    """Run an SQL query and return results in a DataFrame."""
    conn = create_connection(db_file)
    df = pd.read_sql_query(query, conn)
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
