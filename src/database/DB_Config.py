import sqlite3
from typing import Optional, Dict, Any
import psycopg2
from psycopg2 import OperationalError, sql
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_connection(db_name: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> Optional[Any]:
    """
    Create or connect to a database.

    Parameters:
    - db_name (str): Name of the database.
    - host (Optional[str]): Host address (for PostgreSQL).
    - user (Optional[str]): Username (for PostgreSQL).
    - password (Optional[str]): Password (for PostgreSQL).

    Returns:
    - Optional[Any]: Database connection object or None if connection fails.
    """
    try:
        if host:  # PostgreSQL connection
            conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host
            )
            logger.info("Connected to PostgreSQL database.")
        else:  # SQLite connection
            conn = sqlite3.connect(db_name)
            logger.info("Connected to SQLite database.")
        return conn
    except OperationalError as e:
        logger.error(f"Operational error while connecting to the database: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while connecting to the database: {e}")
    return None


def query_database(query: str, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> pd.DataFrame:
    """
    Execute an SQL query and return the results as a DataFrame.

    Parameters:
    - query (str): The SQL query to execute.
    - db_name (str): Name of the database.
    - db_type (str): Type of the database ('sqlite' or 'postgresql').
    - host (Optional[str]): Host address (for PostgreSQL).
    - user (Optional[str]): Username (for PostgreSQL).
    - password (Optional[str]): Password (for PostgreSQL).

    Returns:
    - pd.DataFrame: Query results.
    """
    conn = create_connection(db_name, host, user, password)
    if conn is None:
        logger.error("Database connection failed. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        if db_type.lower() == 'postgresql':
            # Use psycopg2's RealDictCursor for better performance with pandas
            df = pd.read_sql_query(query, conn)
        elif db_type.lower() == 'sqlite':
            df = pd.read_sql_query(query, conn)
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return pd.DataFrame()
        logger.info("Query executed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
        logger.info("Database connection closed.")


def get_all_schemas(db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    conn = create_connection(db_name, host, user, password)
    if conn is None:
        logger.error("Database connection failed. Returning empty schemas.")
        return {}

    cursor = conn.cursor()
    schemas = {}

    try:
        if db_type.lower() == 'sqlite':
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                table_info = {}
                
                # Get column information
                cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
                columns = cursor.fetchall()
                table_info['columns'] = {
                    col[1]: {
                        'type': col[2],
                        'nullable': not col[3],
                        'primary_key': bool(col[5]),
                        'default': col[4]
                    } for col in columns
                }
                
                # Get foreign key constraints
                cursor.execute(f"PRAGMA foreign_key_list(\"{table_name}\");")
                fkeys = cursor.fetchall()
                table_info['foreign_keys'] = [
                    {
                        'from_column': fk[3],
                        'to_table': fk[2],
                        'to_column': fk[4],
                        'on_update': fk[5],
                        'on_delete': fk[6]
                    } for fk in fkeys
                ]
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list(\"{table_name}\");")
                indexes = cursor.fetchall()
                table_info['indexes'] = []
                for idx in indexes:
                    cursor.execute(f"PRAGMA index_info(\"{idx[1]}\");")
                    index_columns = cursor.fetchall()
                    table_info['indexes'].append({
                        'name': idx[1],
                        'unique': bool(idx[2]),
                        'columns': [col[2] for col in index_columns]
                    })
                
                # Get sample data
                cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 5;")
                sample_data = cursor.fetchall()
                if sample_data:
                    column_names = [description[0] for description in cursor.description]
                    table_info['sample_data'] = [
                        dict(zip(column_names, row)) for row in sample_data
                    ]
                
                schemas[table_name] = table_info
                
        elif db_type.lower() == 'postgresql':
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                table_info = {}
                
                # Get column information
                cursor.execute("""
                    SELECT 
                        column_name, 
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length
                    FROM information_schema.columns 
                    WHERE table_name = %s;
                """, [table_name])
                columns = cursor.fetchall()
                table_info['columns'] = {
                    col[0]: {
                        'type': col[1],
                        'nullable': col[2] == 'YES',
                        'default': col[3],
                        'max_length': col[4]
                    } for col in columns
                }
                
                # Get primary key information
                cursor.execute("""
                    SELECT c.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                    JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                        AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                    WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
                """, [table_name])
                pk_columns = cursor.fetchall()
                for col in pk_columns:
                    table_info['columns'][col[0]]['primary_key'] = True
                
                # Get foreign key information
                cursor.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
                """, [table_name])
                fkeys = cursor.fetchall()
                table_info['foreign_keys'] = [
                    {
                        'from_column': fk[0],
                        'to_table': fk[1],
                        'to_column': fk[2]
                    } for fk in fkeys
                ]
                
                # Get indices information
                cursor.execute("""
                    SELECT
                        i.relname as index_name,
                        a.attname as column_name,
                        ix.indisunique as is_unique
                    FROM pg_class t
                    JOIN pg_index ix ON t.oid = ix.indrelid
                    JOIN pg_class i ON i.oid = ix.indexrelid
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                    WHERE t.relkind = 'r' AND t.relname = %s;
                """, [table_name])
                indexes = cursor.fetchall()
                table_info['indexes'] = []
                current_index = None
                for idx in indexes:
                    if current_index is None or current_index['name'] != idx[0]:
                        if current_index is not None:
                            table_info['indexes'].append(current_index)
                        current_index = {'name': idx[0], 'unique': idx[2], 'columns': []}
                    current_index['columns'].append(idx[1])
                if current_index is not None:
                    table_info['indexes'].append(current_index)
                
                # Get sample data
                cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 5;")
                sample_data = cursor.fetchall()
                if sample_data:
                    column_names = [description[0] for description in cursor.description]
                    table_info['sample_data'] = [
                        dict(zip(column_names, row)) for row in sample_data
                    ]
                
                schemas[table_name] = table_info

        logger.info("Enhanced schema information retrieved successfully.")
        return schemas

    except Exception as e:
        logger.error(f"Error retrieving enhanced schema information: {e}")
        return {}
    finally:
        conn.close()
        logger.info("Database connection closed.")

    return schemas
