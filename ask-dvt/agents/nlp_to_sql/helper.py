import re, os, io, sys, json, logging, textwrap
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
import pandas as pd
from state import ToolBaseSchema
from sqlalchemy import create_engine
from configuration import Configuration
from typing import List, Dict, Any, Optional
from llama_index.core.llms import ChatResponse
from llama_index.core.retrievers import SQLRetriever
from pydantic import BaseModel, Field, ValidationError
from llama_index.core.utils import set_global_tokenizer
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent, FnComponent
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema

class NLPToSQLToolInput(ToolBaseSchema):
    pass

class AnalysisResult(BaseModel):
    summary: str = Field(description="A summary of the query results.")
    insights: list[str] = Field(description="Key insights extracted from the query results.")

class LoggingSQLRetriever(SQLRetriever):
    def __init__(self, sql_database, **kwargs):
        super().__init__(sql_database, **kwargs)
        self.last_sql_query = None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        self.last_sql_query = query_bundle.query_str
        return super()._retrieve(query_bundle)

    def get_last_sql_query(self) -> str:
        return self.last_sql_query

class NLPToSQLHelper:
    def __init__(self):
        self.DB_HOST = os.environ.get("DB_HOST")
        self.DB_NAME = "dvt-db"
        self.DB_USER = os.environ.get("DB_USER")
        self.DB_PASSWORD = os.environ.get("DB_PASSWORD")
        self.DB_PORT = int(os.environ.get("DB_PORT"))
        
        self.llm = Configuration.query_model
        self.embedding_model = Configuration.embedding_model
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        set_global_tokenizer(NLPToSQLHelper.simple_token_counter)       
        self.initialize_database_engine()
        self.create_pipeline_components()
    
    @staticmethod
    def simple_token_counter(text: str) -> int:
        return len(text)
    
    def initialize_database_engine(self):
        DATABASE_URL = f"""postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"""
        try:
            engine = create_engine(DATABASE_URL)
            self.sql_database = SQLDatabase(engine, include_tables=["stat_data", "execution_data"])
            self.sql_retriever = LoggingSQLRetriever(self.sql_database)
        except Exception as e:
            logging.error(f"Error Connecting to SQL Database: {e}", exc_info=True)
    
    def parse_response_to_sql(self, response: ChatResponse) -> str:
        """Parse response to SQL, removing markdown delimiters."""
        response_content = response.message.content.lower()
        response_content = re.sub(r"```(sql)?", "", response_content).strip() #Remove code block delimiters
        response_content = re.sub(r"```", "", response_content).strip() #Remove code block delimiters
        sql_query_start = response_content.find("sqlquery:")

        if sql_query_start != -1:
            sql_query_text = response_content[sql_query_start + len("sqlquery:"):]
            # Remove "sqlresult" and anything after it.
            sql_result_start = sql_query_text.lower().find("sqlresult:")
            if sql_result_start != -1:
                sql_query_text = sql_query_text[:sql_result_start]
                sql_query_text = sql_query_text.strip()
                return sql_query_text
        elif len(response_content) > 0:
            return response_content
        else:
            return "" #Return empty string if no sql query is found.

    def get_table_context_str(self, table_schema_objs: List[SQLTableSchema]):
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = self.sql_database.get_single_table_info(table_schema_obj.table_name)
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context
            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    def create_pipeline_components(self):
        try:
            # Table schema definitions
            table_node_mapping = SQLTableNodeMapping(self.sql_database)
            table_schema_objs = [
                SQLTableSchema(table_name="stat_data",
                                context_str="This is metrics table. column id is PK. column or field statvalue represents record count many to one join: stat_data.global_execution_id = execution_data.id and stat_data.global_profile_instance = execution_data.profile_instance"),
                SQLTableSchema(table_name="execution_data",
                            context_str="""
                                * This table contains one record for every job or profile execution.
                                * Combination of columns id and profile_instance is PK.
                                * column or field profile_instance is unique identifier for a job.
                                * Time difference between start_time and end_time is job run or execution time. 
                                * Column or field status field represents overall status of a job
                            """)
            ]

            # Object index and retriever
            obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
            
            self.obj_retriever = obj_index.as_retriever(similarity_top_k=5)        
            self.table_parser_component = FnComponent(fn=self.get_table_context_str)
            self.sql_parser_component = FnComponent(fn=self.parse_response_to_sql)
            self.text2sql_prompt = PromptTemplate(self.get_txt_to_sql_prompt())
            
            response_synthesis_prompt_str = (
                "Given an input question, synthesize a response from the query results.\n"
                "Query: {query_str}\n"
                "SQL: {sql_query}\n"
                "SQL Summary Response: {context_str_summary}\n"
                "SQL Response: {context_str}\n"
                "Response: "
            )
            self.response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
        except Exception as e:
            logging.error(f"Error Creating Query Pipeline: {e}", exc_info=True)
        
    def create_e2e_pipeline(self):
        try:
            qp = QP(
                modules={
                    "input": InputComponent(),
                    "table_retriever": self.obj_retriever,
                    "table_output_parser": self.table_parser_component,
                    "text2sql_prompt": self.text2sql_prompt,
                    "text2sql_llm": Settings.llm,
                    "sql_output_parser": self.sql_parser_component,
                    "sql_retriever": self.sql_retriever,
                    "response_synthesis_prompt": self.response_synthesis_prompt,
                    "response_synthesis_llm": Settings.llm,
                },
                verbose=False,
            )

            qp.add_chain(["input", "table_retriever", "table_output_parser"])
            qp.add_link("input", "text2sql_prompt", dest_key="query_str")
            qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
            qp.add_chain(["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"])
            qp.add_link("sql_output_parser", "response_synthesis_prompt", dest_key="sql_query")
            qp.add_link("sql_retriever", "response_synthesis_prompt", dest_key="context_str")
            qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
            qp.add_link("response_synthesis_prompt", "response_synthesis_llm")
            
            return qp
        except Exception as e:
            logging.error(f"Error Creating Query Pipeline: {e}", exc_info=True)
            return None
   
    def create_querygen_pipeline(self):
        qp = QP(
            modules={
                "input": InputComponent(),
                "table_retriever": self.obj_retriever,
                "table_output_parser": self.table_parser_component,
                "text2sql_prompt": self.text2sql_prompt,
                "text2sql_llm": Settings.llm,
                "sql_output_parser": self.sql_parser_component,
            },
            verbose=False,
        )

        qp.add_chain(["input", "table_retriever", "table_output_parser"])
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        qp.add_chain(["text2sql_prompt", "text2sql_llm", "sql_output_parser"])

        return qp
   
    def create_queryexec_pipeline(self):
        qp = QP(
            modules={
                "input": InputComponent(), #Or create a new input component that directly accepts sql
                "sql_retriever": self.sql_retriever,
                "response_synthesis_prompt": self.response_synthesis_prompt,
                "response_synthesis_llm": Settings.llm,
            },
            verbose=False,
        )

        qp.add_chain(["input", "sql_retriever"])
        qp.add_link("input", "sql_retriever")
        qp.add_link("sql_retriever", "response_synthesis_prompt", dest_key="context_str")
        qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

        return qp
   
    def get_database_engine(self):
        return self.sql_database
    
    def get_sql_retriever(self):
        return self.sql_retriever

    def get_last_query(self):
        return self.sql_retriever.get_last_sql_query()
    
    def get_llm(self):
        return self.llm
    
    def get_embedding_model(self):
        return self.embedding_model

    def get_agent_prompt(self):
        return Configuration.nlp_to_sql_agent_prompt

    def get_txt_to_sql_prompt(self):
        return Configuration.text_to_psql_converter_prompt
        
    def get_data_analyst_prompt(self):   
        return Configuration.data_analysis_expert_prompt   

    def extract_dataframe_from_text(self, text):
        """
        Extracts a pandas DataFrame from a string containing natural language text and a CSV dataset.

        Args:
            text (str): The input string containing natural language text and a CSV dataset.

        Returns:
            csv_data (str): Valid CSV data if found, or None if not found.
        """
        try:
            # Split the text into lines
            lines = text.splitlines()

            # Find the start of the CSV data (the header)
            csv_start_index = None
            for i, line in enumerate(lines):
                if ',' in line:  # Check for comma-separated values (CSV)
                    csv_start_index = i
                    break

            if csv_start_index is None:
                return None  # No CSV data found

            # Extract the CSV data
            csv_data = '\n'.join(lines[csv_start_index:])

            # Create a DataFrame from the CSV data
            df = pd.read_csv(io.StringIO(csv_data))

            # Check if the DataFrame has at least one row
            if len(df) > 0:
                return csv_data
            else:
                return None  # DataFrame is empty

        except Exception as e:
            print(f"Error extracting DataFrame: {e}")
            return None


# print(RAGHelper.get_query_embedding("Hello, world!"))