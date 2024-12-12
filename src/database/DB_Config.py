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


def get_all_schemas(db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Retrieve schema information for all tables in the database.

    Parameters:
    - db_name (str): Name of the database.
    - db_type (str): Type of the database ('sqlite' or 'postgresql').
    - host (Optional[str]): Host address (for PostgreSQL).
    - user (Optional[str]): Username (for PostgreSQL).
    - password (Optional[str]): Password (for PostgreSQL).

    Returns:
    - Dict[str, Dict[str, str]]: Schema details for each table.
    """
    conn = create_connection(db_name, host, user, password)
    if conn is None:
        logger.error("Database connection failed. Returning empty schemas.")
        return {}

    cursor = conn.cursor()
    schemas = {}

    try:
        if db_type.lower() == 'sqlite':
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
                columns = cursor.fetchall()
                schema = {col[1]: col[2] for col in columns}
                schemas[table_name] = schema
        elif db_type.lower() == 'postgresql':
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                cursor.execute(sql.SQL("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s;
                """), [table_name])
                columns = cursor.fetchall()
                schema = {col[0]: col[1] for col in columns}
                schemas[table_name] = schema
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return schemas

        logger.info("Schemas retrieved successfully.")
    except Exception as e:
        logger.error(f"Error retrieving schemas: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed.")

    return schemas
