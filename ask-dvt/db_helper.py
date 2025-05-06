import os, psycopg2
from pgvector.psycopg2 import register_vector

class pgdbWrapper:
    """Helper class to manage connection to postgres database"""
    # DB_HOST = "host.docker.internal"

    def __init__(self, db_host=None, db_name=None, db_user=None, db_password=None, db_port=None):
        self.DB_HOST = db_host or os.environ.get("DB_HOST")
        self.DB_NAME = db_name or os.environ.get("DB_NAME")
        self.DB_USER = db_user or os.environ.get("DB_USER")
        self.DB_PASSWORD = db_password or os.environ.get("DB_PASSWORD")
        self.DB_PORT = db_port or os.environ.get("DB_PORT")
        self.connection, self.cursor = None, None
        
    def get_db_credentials(self):
        return (self.DB_HOST, self.DB_NAME, self.DB_USER, self.DB_PASSWORD, self.DB_PORT)
    
    def connect(self):
        self.connection = psycopg2.connect(
            host=self.DB_HOST,
            database=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            port=self.DB_PORT  # CRITICAL: Using self.DB_PORT
        )
        register_vector(self.connection)
        self.cursor = self.connection.cursor()
    
    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            
    def execute_select_query(self, query:str, params=()):      
        if not self.connection: self.connect()
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        return results
    
    def execute_insert_or_update(self, query: str, params=()):
        if not self.connection:
            self.connect()
        try:
            # print(self.cursor.mogrify(query, params).decode('utf-8'))
            self.cursor.execute(query, params)
            self.connection.commit()  # Commit the transaction

            # Check if the query was an INSERT with a RETURNING clause
            if query.lower().startswith("insert") and "returning" in query.lower():
                returned_value = self.cursor.fetchone()
                if returned_value:
                    return returned_value[0]  # Return the first column of the first row
                else:
                    return True  # INSERT was successful, but no value returned (unlikely with RETURNING)
            else:
                return True  # It was an UPDATE or an INSERT without RETURNING
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()  # Rollback on error
            print(f"Error executing INSERT/UPDATE query: {e}")
            return False
    
    def execute_bulk_insert(self, query: str, params_list: list):
         if not self.connection: self.connect()
         try:
            self.cursor.executemany(query, params_list)
            self.connection.commit()
            return True
         except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            print(f"Error executing bulk insert query: {e}")
            return False