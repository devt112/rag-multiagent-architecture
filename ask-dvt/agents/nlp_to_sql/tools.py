import re, os, sys, json, logging, textwrap, urllib3
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
import pandas as pd
from typing import Optional, Dict, Any
from agents.nlp_to_sql.helper import NLPToSQLHelper, NLPToSQLToolInput
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain.tools import StructuredTool, tool
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

helper = NLPToSQLHelper()

@tool(
    "Generate_SQL_Query_from_Natural_Language_and_Execute_Query",
    description = textwrap.dedent(f"""
    This tool translates natural language questions into SQL queries and executes them against the company's internal database. Use this tool to translate user questions or requests into syntactically correct SQL queries for retrieving structured data from internal database.

    **When to Use:**
    -   When the user asks a question that requires retrieving data from a database.
    -   When you need to translate user intent into specific SQL commands to query a database.
    -   When the user is asking for data retrieval, filtering, aggregation, or other database operations using natural language.
    -   Use this tool to generate data for reports, analysis, or further processing.
    -   When the user provides a natural language query and expects a table or structured data as a result.
    -   When a user requests a pivot table, or other complex data manipulation.
    -   When the query involves finding counts, sums, averages, or other aggregations.
    -   When the query involves date/time calculations or comparisons.
    -   When the query requires finding maximum or minimum values (e.g., "latest date").

    **Example Usage:**
    -   "List the top 10 jobs by input record count."
    -   "Get the job status for profile instance '20240924202417278457471-d4y1d'."
    -   "Find FAILED jobs after January 1, 2023."
    -   "Retrieve the average job execution time in minutes for January 2025."
    -   "What is the total number of jobs execute this year?"
    -   "Which customers have maximum failed jobs?"
    -   "Give me a report of daily count of COMPLETE jobs with average execution time in minutes from last 15 days from today"
    -   "Tell me the all the details about job with profile instance '20230801082934463'"
    -   "Provide a list of all RUNNING jobs."
    -   "Display all the step names for profile instance '20240814093754894248088-1ut3j'."
    -   "Extract the job history for month February 2025"
    -   "Number of failed jobs compared to last quarter."
    -   "Jobs with execution time greater than 5 hours."
    -   "Jobs with number of input records less than 1,000,000."
    -   "Number of jobs failed between January 2025 and March 2025"
    -   "Profiles with stepname like 'csv-avro%'."
    -   "Show results for the query: monthly percentage job count split by status from last 12 months from max date as a table sorted by date desc."
    -   "How many jobs FAILED in 24 hours calculated from the most recent date available?"
    -   "Show list of profile instances of jobs that FAILED in 24 hours calculated from the most recent start date available."
    -   "Generate a pivot table with the following structure: Rows: Month (in descending order), Columns: Job Status, Values: Percentage of jobs for each status within each month. The data should cover the last 10 months, calculated from the most recent date available."

    **When NOT to Use:**
    -   Do NOT use this tool if the user provides data directly (e.g., CSV) and asks for analysis (use Analyze_Data_Extract_Insights).
    -   Do NOT use this tool for general information retrieval that doesn't involve a database query.
    -   Do NOT use this tool if the user is asking for a summary or interpretation of data (use Analyze_Data_Extract_Insights *after* this tool).

    **Input:**
    -   `query` (string, required): The user's question or request in natural language.

    **Output:**
    -   A dictionary containing the original user input (as query), the generated sql query (as code), the query results in CSV format (as datatable), an empty string for data analysis (as response) and additional info.
    """),
    args_schema = NLPToSQLToolInput,
    return_direct = False   
)
def generate_execute_sql_query(query: str, config: RunnableConfig) -> dict:
    """
    Generates a SQL query from a user's natural language input string,
    executes the generated query, and returns the result in CSV format.

    This function takes a natural language query, transforms it into a SQL query,
    executes that query against the database, and formats the results as a CSV string.

    Args:
        query (str): The user's natural language query for database interaction.

    Returns:
        dict: A dictionary containing the results of the SQL query execution. The dictionary has the following keys:

        -   'query' (str): The original user's natural language query. This is the initial input provided to the function.
        -   'code' (str): The SQL query generated from the natural language query. This is the query that was executed against the database.
        -   'datatable' (str): The query result formatted as a CSV string. If the query returns data, this string will include the column headers in the first row, followed by the data rows. If the query returns no data, this may be an empty string or a CSV with only headers.
        -   'response' (str):  An empty string. This field is intended for use by subsequent tools that might analyze the data. This tool itself does not perform analysis.
        -   'info' (dict):  Additional information or context related to the response.

        Example of a successful response:
        {
            'query': 'Show me all customers from California',
            'code': 'SELECT * FROM customers WHERE state = \'CA\'',
            'datatable': 'customer_id,name,city,state\n1,John Doe,Los Angeles,CA\n2,Jane Smith,San Francisco,CA\n',
            'response': '',
            'info': {'_info': 'This is a sample response'}
        }

    Raises:
        Exception: If an error occurs during query generation or execution.
    """
    print("===============================NLP TO SQL===============================")

    result = NLPToSQLToolInput(query=query, code="", datatable="", response="", info={})
    
    try:
        query_pipeline = helper.create_e2e_pipeline()
        if query_pipeline is not None:
            node_with_score = query_pipeline.run(query=query)
            if node_with_score and node_with_score[0].node.metadata:
                metadata = node_with_score[0].node.metadata
                datatable = metadata.get('result')
                col_keys = metadata.get('col_keys')
                if datatable and col_keys:
                    result.code = metadata.get('sql_query')
                    result.datatable = pd.DataFrame(datatable, columns=col_keys).to_csv(index=False)
    except Exception as e:
        logging.error(f"Error executing SQL query: {e}", exc_info=True)
    return result.model_dump()

# --------------------------------------------------------------------------------

@tool(
    "Analyze_Data_Extract_Insights",
    description = textwrap.dedent(f"""
    This tool analyzes structured data, such as database query results or CSV content, to generate human-readable summaries and identify key insights. It excels at uncovering patterns, trends, and significant findings within the data, facilitating deeper understanding and informed decision-making.

    **Key Capabilities:**

    -   **Data Summarization:** Provides concise summaries of complex datasets.
    -   **Pattern Recognition:** Identifies recurring patterns and anomalies.
    -   **Trend Analysis:** Highlights changes and trends over time or categories.
    -   **Insight Extraction:** Draws meaningful conclusions from the data.
    -   **Natural Language Reporting:** Presents findings in an easy-to-understand format.

    **When to Use:**

    -   **Post-SQL Analysis:** This tool is PRIMARILY designed to analyze data *RETRIEVED FROM A DATABASE* using the 'Generate_SQL_Query_from_Natural_Language_and_Execute_Query' tool. Use it to analyze the results of a database query.
    -   **Direct Data Analysis:** You can also use this tool when the user provides data directly (e.g., CSV content) and requests analysis.
    -   **Data Interpretation:** When the user needs help understanding the meaning and implications of data.
    -   **Report Generation:** To create human-readable reports from data.
    -   **Data Exploration:** When the user wants to explore and understand data characteristics and relationships.

    **When NOT to Use:**

    -   **Data Retrieval:** Do NOT use this tool to retrieve data from a database. Use 'Generate_SQL_Query_from_Natural_Language_and_Execute_Query' for that purpose.
    -   **Direct Question Answering:** Do NOT use this tool to answer simple questions that can be directly answered from the database (e.g., "What is the average salary?"). Use 'Generate_SQL_Query_from_Natural_Language_and_Execute_Query' to get the answer.
    -   **Do NOT use this tool as the FIRST step if the data needs to be fetched from the database. Always use 'Generate_SQL_Query_from_Natural_Language_and_Execute_Query' first.**

    **Input:**

    -   `query` (str): The user's original question or context that led to the data analysis.  This provides context for the analysis.
    -   `code` (str, optional): The SQL query (if any) that was used to generate the `datatable`. Providing the query can help the tool understand the data's origin and meaning. Defaults to None.
    -   `datatable` (str, optional): The data to be analyzed, typically in CSV format.  Defaults to None.

    **Output:**

    -   A dictionary containing the data analysis results:
        -   `query` (str): The original user-provided query or context.
        -   `code` (str, optional): The SQL query used to generate the data (if applicable).
        -   `datatable` (str, optional): The analyzed data (if applicable).
        -   `response` (str): The natural language analysis and insights generated by the LLM. This is the key output.
        -   `info` (dict):  (Currently None in the provided code, but reserved for potential future metadata about the analysis).
    """),
    args_schema = NLPToSQLToolInput,
    return_direct = False
)
def generate_data_insights(query: str, code: str = None, datatable: str = None, config: RunnableConfig = {"configurable": {}}) -> dict:
    """
    Analyzes data using a Large Language Model (LLM) to generate human-readable insights and summaries.

    This function takes data (optionally with the originating SQL query) and uses an LLM to provide a descriptive analysis.

    Args:
        query (str): The user's natural language query or context that prompted the data analysis.
        code (str, optional): The SQL query that was used to generate the 'datatable', if applicable. Defaults to None.
        datatable (str, optional): The data to be analyzed, typically in CSV format. Defaults to None.

    Returns:
        dict: A dictionary containing the analysis results:
        -   `query` (str): The original user query.
        -   `code` (str, optional): The SQL query (if provided).
        -   `datatable` (str, optional): The analyzed data (if provided).
        -   `response` (str): The LLM-generated natural language analysis of the data.
        -   `info` (dict):  (Currently None, reserved for future metadata).

    Example:
    ```
    {
        'query': 'Analyze sales data for the last quarter',
        'code': 'SELECT * FROM sales WHERE quarter = 4 AND year = 2023',
        'datatable': 'product,revenue\\nA,12000\\nB,15000\\nC,9000\\n',
        'response': 'Sales for the last quarter show that Product B generated the highest revenue (15000), while Product C generated the lowest (9000). Overall revenue was 36000.',
        'info': {'_info': 'This is a sample response'}
    }
    ```
    """
    print("===============================DATA ANALYSIS===============================") 
    result = NLPToSQLToolInput(query=query, code=code, datatable=datatable, response="", info={})

    llm_input = None
    if query and datatable:
        llm_input = "Context: " + query + "\n\nQuestion: Analyze the dataset below:\n" + datatable
    elif query and not datatable:
        llm_input = "Context: " + query
        result.datatable = helper.extract_dataframe_from_text(query)
    elif not query and datatable:
        llm_input = "Context: Analyze the dataset below:\n" + datatable
    else: return result.model_dump()
    
    llm_input = llm_input
    llm = helper.get_llm()
    agent_prompt = PromptTemplate.from_template(helper.get_data_analyst_prompt())
    chain = agent_prompt | llm
    analysis = chain.invoke({"input": llm_input})
    result.response = analysis.content
    return result.model_dump()

# --------------------------------------------------------------------------------