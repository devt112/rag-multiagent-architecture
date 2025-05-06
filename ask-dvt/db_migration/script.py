import os
import sys
import warnings
import time
import json
import markdown
import re
import urllib3

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from db_helper import pgdbWrapper

source_db = pgdbWrapper()
dest_db = pgdbWrapper()


def get_all_tables(db: pgdbWrapper):
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'  -- You can change 'public' to the desired schema
        AND table_type = 'BASE TABLE';
    """
    results = db.execute_select_query(query)
    tables = [row[0] for row in results]
    return tables


def get_column_names(db: pgdbWrapper, table_name: str, schema_name: str = 'public') -> list[str]:
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        AND table_schema = %s
    """
    params = (table_name, schema_name)
    results = db.execute_select_query(query, params)
    return results


def copy_table_data(source_db: pgdbWrapper, dest_db: pgdbWrapper, table_name: str,
                    schema_name: str = 'public'):
    # 1. Get column names (and types, if needed for more complex logic)
    columns_info = get_column_names(source_db, table_name, schema_name)
    columns = [col[0] for col in columns_info if col[0] != 'id']  # Skip 'id'
    columns_str = ", ".join(columns)
    placeholders = ", ".join(['%s'] * len(columns))  # Create placeholders for insert

    # 2. Build SELECT query for source (excluding 'id')
    select_query = f"SELECT {columns_str} FROM {schema_name}.{table_name}"

    # 3. Build INSERT query for destination
    insert_query = f"INSERT INTO {schema_name}.{table_name} ({columns_str}) VALUES ({placeholders})"

    # 4. Fetch data from source
    source_data = source_db.execute_select_query(select_query)

    # 5. Insert data into destination (using individual inserts)
    if source_data:
        for row in source_data:
            dest_db.execute_insert_or_update(insert_query, row)  # Insert one row at a time
        print(f"Copied {len(source_data)} rows from {schema_name}.{table_name} (excluding id)")
    else:
        print(f"No data to copy from {schema_name}.{table_name}")


if __name__ == '__main__':
    try:
        tables = get_all_tables(source_db)
        for table in tables:
            print(f"Copying data from table: {table}")
            copy_table_data(source_db, dest_db, table, "public")  # Assuming 'public' schema
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        source_db.disconnect()
        dest_db.disconnect()