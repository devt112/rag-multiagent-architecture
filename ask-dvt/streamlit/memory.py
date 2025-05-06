import os, sys, re, json, urllib3, uuid, time, logging
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
import streamlit as st
import concurrent.futures
from datetime import datetime
from db_helper import pgdbWrapper
from commons.logger import CustomLogger
# import extra_streamlit_components as stx
from commons.processing import Processor
# from idx_db_component.frontend import idxdb_component


class MemoryManager:
    """
    Manages session state and database interactions for the Streamlit app.
    This class is responsible for initializing, storing, loading, and clearing
    data, as well as handling the database connection.
    """
    def __init__(self):
        self.db = pgdbWrapper()
        self.init_db()
        self.processor = Processor()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        my_logger = CustomLogger(
            level=logging.DEBUG,
            log_format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
        self.logger = my_logger.get_logger()

    def get_thread_history(self):        
        query = """
            with cte as (
                SELECT ROW_NUMBER() OVER (PARTITION BY ch.session_id, ch.thread_id ORDER BY created DESC, timestamp DESC) AS rn,
                cs.session_id, cs.thread_id, 
                CASE WHEN LENGTH(ch.content) > 35 THEN LEFT(TRIM(REPLACE(ch.content, E'\n', ' ')), 35) || '...'
                ELSE COALESCE(TRIM(REPLACE(ch.content, E'\n', ' ')),'') END AS latest_user_content,
                TO_CHAR(cs.created::DATE, 'MM-DD-YYYY') AS created_date
                FROM chat_sessions as cs
                LEFT JOIN (SELECT * FROM chat_history WHERE role='user') as ch
                ON cs.session_id=ch.session_id AND cs.thread_id=ch.thread_id
                WHERE cs.session_id = %s
            )
            SELECT session_id, thread_id, latest_user_content, created_date
            FROM cte WHERE rn=1 ORDER BY thread_id DESC LIMIT 11;
        """
        params = (st.session_state.session_id,)
        # print("Getting thread history...", st.session_state.session_id)
        result = self.db.execute_select_query(query, params)
        # print("Thread history result:", result)
        return result
    
    @staticmethod
    @st.cache_data
    def get_cookie():
        sid, counter, max_counter = None, 1, 3
        while sid is None and counter <= max_counter:
            try:
                sid = st.context.cookies["ajs_anonymous_id"]
            except:
                self.logger.error("Error Retrieving ajs_anonymous_id from cookies.")
                time.sleep(5)
                counter = counter + 1
        return sid
    
    def get_session_id(self) -> str:
        sid = MemoryManager.get_cookie()
        if sid is not None:
            query = f"""
                INSERT INTO public.chat_sessions (session_id, thread_id)
                SELECT
                    '{sid}',
                    COALESCE(
                        (SELECT MAX(cs.thread_id) + 1
                        FROM public.chat_sessions cs
                        WHERE cs.session_id = '{sid}'),
                        1
                    )
                WHERE EXISTS (
                    SELECT 1
                    FROM public.chat_history ch
                    WHERE ch.session_id = '{sid}'
                    AND ch.thread_id = COALESCE(
                        (SELECT MAX(cs2.thread_id) 
                        FROM public.chat_sessions cs2 
                        WHERE cs2.session_id = '{sid}'), 0)
                )
                OR NOT EXISTS (SELECT 1 FROM public.chat_sessions WHERE session_id = '{sid}')
                RETURNING thread_id;
            """
            new_thread_id = self.db.execute_insert_or_update(query, None)
            st.session_state.session_id = sid
            st.session_state.session_id = sid
            self.get_thread_history()
        return sid
    
    def initialize_session_state(self):
        """Initializes session state variables if not already present."""
        if "stage" not in st.session_state:
            st.session_state.stage = "user"
            st.session_state.history = []
            st.session_state.pending = None
            st.session_state.validation = {} 
            st.session_state.chat_input_disabled = False
            st.session_state.user_config = {"configurable":{"thread_id": str(uuid.uuid4())}}
            self.get_session_id()
            # print("Session state initialized:", st.session_state)

    def clear_session_state_vars(self, keys_to_preserve=None, clear_history=False, restart=False):
        """
        Clears session state variables, preserving specified keys.
        """
        if keys_to_preserve is None:
            keys_to_preserve = ["history", "agent_state", "agent_response", "memory_manager", "init", "set", "session_id", "stage", "session_history", "chat_input_disabled", "user_config"]

        for key in st.session_state.keys():
            if key not in keys_to_preserve:
                del st.session_state[key]

        if "history" in st.session_state and "pending" in st.session_state:
            st.session_state.history.append({
                "role": "assistant",
                "content": {
                    "response_time": st.session_state.get("response_time"),
                    "pending": st.session_state.get("pending")
                }
            })
        st.session_state["validation"] = {}
        st.session_state["stage"] = "user"
        if restart:
            st.rerun()

    def init_db(self):
        """Initializes the PostgreSQL database connection."""
        
        sql_query = """
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                queid TEXT,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """
        return self.db.execute_insert_or_update(sql_query)

    def store_message(self, session_id, role, content):
        query = """
            INSERT INTO chat_history (session_id, role, content)
            VALUES (%s, %s, %s)
        """
        params = (session_id, role, content)
        # Submit the database operation to the thread pool
        future = self.executor.submit(self.db.execute_insert_or_update, query, params)
        return future  # Return the Future object
    
    def store_messages(self, messages:[dict]):
        def prepare_bulk_insert_data(data: list[dict]) -> list[tuple]:
            expected_keys = ['session_id', 'role', 'content', 'queid', 'thread_id']
            params_list = []
            for row in data:
                if all(key in row for key in expected_keys):
                    params_list.append((row['session_id'], row['role'], row['content'], row['queid'], row['thread_id']))
                else:
                    print(f"Skipping row: {row}.  It does not contain all required keys: {expected_keys}")
            return params_list
        
        query = """
            INSERT INTO chat_history (session_id, role, content, queid, thread_id)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        params_list = prepare_bulk_insert_data(messages)
        future = self.executor.submit(self.db.execute_bulk_insert, query, params_list)
        return future  # Return the Future object

    def load_history(self, session_id, thread_id=None):
        if thread_id is None:
            query = """
                with cte as (
                    SELECT session_id, MAX(thread_id) as thread_id
                    FROM chat_sessions
                    WHERE session_id=%s
                    GROUP BY session_id
                )
                SELECT role, content
                FROM cte LEFT JOIN chat_history as ch
                ON cte.session_id = ch.session_id AND cte.thread_id = ch.thread_id
                ORDER BY timestamp, role desc
                LIMIT 50;
            """
            params = [session_id]
        else:
            query = """
                SELECT role, content
                FROM chat_history
                WHERE session_id = %s AND thread_id = %s
                ORDER BY timestamp, role desc
                LIMIT 50;
            """
            params = (session_id, thread_id)
        future = self.executor.submit(self.db.execute_select_query, query, tuple(params))
        return future  # Return the Future object

    def create_new_session(self, session_id=None):
        session_id = self.get_session_id()
        st.session_state.history = []
        st.session_state.stage = "user"
        st.session_state.pending = None
        st.session_state.validation = {}
        return True
