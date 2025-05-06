import os, sys, re, json, urllib3, textwrap
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from db_helper import pgdbWrapper
from commons.utils import Utilities
from agents.rag_kb_agent.helper import RAGHelper, RAGInput
from typing_extensions import Annotated
from langchain.tools import StructuredTool, tool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

helper = RAGHelper()

def generate_llm_response(query, context):
    prompt = textwrap.dedent(f"""
        You are a helpful and informative assistant. Your task is to answer questions based on the provided context and additionally extract any associated runbook metadata that follows a recognizable structure.

        **Recognizable Runbook Metadata Structure:** Runbook metadata, when present, often follows a YAML-like structure denoted by `---` at the beginning and end and *may* include some or all of the following fields:

        - `process_name`: The name of the process the runbook relates to.
        - `alias`: A shorter or alternative name for the process.
        - `process_description`: A brief description of the runbook's purpose.
        - `process_page_url`: A link to a relevant resource.
        - `has_runbook`: A boolean value indicating if a runbook file exists.
        - `runbook_name`: The name of the runbook file.
        - `tasks_covered`: A list of specific tasks covered by the runbook.

        The presence and completeness of these fields may vary.

        Instructions:
        1. **Answer in Detail:** Provide a comprehensive answer to the question, explaining the reasoning behind your response. Don't just give a one-sentence answer.
        2. **Extract Runbook Metadata:** If the provided context contains information that appears to be runbook metadata (enclosed in `---` and containing one or more of the recognizable fields), extract each instance of this metadata. Extract all the fields that are present in each metadata block.
        3. **Code Sample (If Applicable):** If the question requires or would benefit from a code sample, provide a relevant example in a suitable programming language. Choose the language that best fits the context of the question or the user's needs. If no code sample is needed, skip this section.
        4. **Troubleshooting Steps (If Applicable):** If the question relates to a technical issue, suggest potential solutions or troubleshooting steps. Be specific and provide clear instructions. If no troubleshooting is needed, skip this section.
        5. **Confluence/Jira Links (If Applicable):** If you know of any relevant Confluence pages or Jira tickets that could help the user, include the links in your response. If no links are available, skip this section.
        6. **If No Answer:** If the context does not contain the answer, say "I don't know" or "The answer is not in the provided information." Do NOT hallucinate or make up information.
        7. **Clarity and JSON Output:** CRUCIALLY, You MUST Format your response as a JSON object with two keys: "answer" and "runbook_metadata". The "answer" key should contain the detailed answer to the user's query in a clear and easy-to-read manner, using headings, subheadings, bullet points, and numbered lists where appropriate, and the "runbook_metadata" key should contain a list of any extracted runbook metadata. Each extracted metadata block should be represented as a dictionary containing only the fields that were found. If no runbook metadata is found, the "runbook_metadata" key should contain an empty list (`[]`). 

        Context:
        {context}
        Question:
        {query}
    """)
    response = helper.get_llm().invoke(prompt)
    return response.content

#--------------------------------------------------------------------------------------------
@tool(
    "Search_Knowledge_Base",
    description=f"""
    **Key Feature:**
    This tool searches for information *exclusively* within an internal knowledge base of documents.

    **When to Use:**
    -   Use this tool when the user's query requires detailed information or specific facts that are likely to be found within the knowledge base documents.
    -   Use this tool when up-to-date information from internal documentation is crucial.
    -   Use this tool when the query requires searching the raw knowledge base content, not pre-approved answers.

    **When NOT to Use:**
    -   Do NOT use this tool if the user needs a concise, validated answer.
    -   Do NOT use this tool if the user is seeking a summary of information from multiple sources.

    **Input Parameters:**
    -   `query` (string): The user's natural language query. This parameter is *mandatory*.

    **Output:**
    -   A natural language response generated from the retrieved knowledge base documents.
    """,
    args_schema= RAGInput,
    return_direct= False
)
def kb_search_tool(query: str, config:RunnableConfig) -> dict:
    """
    Performs semantic search over an internal knowledge base to answer user queries.

    This function leverages vector embeddings to find relevant document chunks within a vector database
    based on the semantic similarity to the user's query. The retrieved chunks are then concatenated and provided
    as context to a Large Language Model (LLM) to generate a natural language response.

    Key Characteristics:
    - **Semantic Search:** Utilizes vector embeddings and similarity metrics to identify relevant information.
    - **Raw Data Retrieval:** Directly queries a vector database of unstructured documentation.
    - **LLM Contextualization:** Provides retrieved document chunks as context to the LLM for dynamic response generation.
    - **Document Linking:** Attempts to provide direct links to the source documents (Confluence or Jira) when available.
    - **Structured Output Handling:** Expects and attempts to parse the LLM response for a structured dictionary,
      specifically looking for an 'answer' key and potentially 'runbook_metadata'.

    Args:
        query (str): The user's natural language question or prompt.
        config (RunnableConfig): A configuration object that may contain parameters for the tool,
                                 such as the number of similar documents to retrieve ('num_similar_docs').

    Returns:
        dict: A dictionary containing the results of the knowledge base search and LLM generation:
            - 'query' (str): The original user input.
            - 'info' (dict): A dictionary providing details about the tool's execution:
                - '_info' (str): A human-readable description of the tool's operation and parameters,
                  including the number of similar documents retrieved and links to referenced documents.
                - 'is_preapproved' (bool): Indicates if the response is based on pre-approved content (always False in this tool).
                - 'runbook_metadata' (list[dict] or None): If the LLM response includes structured runbook metadata,
                  it will be present here as a list of dictionaries. Otherwise, it will be None.
            - 'response' (str): The natural language answer generated by the LLM based on the retrieved context.
              This will be the raw LLM response if it cannot be parsed into a dictionary with an 'answer' key.
            - 'code' (str or None): Reserved for code snippets generated by the LLM (currently None).
            - 'datatable' (str or None): Reserved for data tables generated by the LLM (currently None).

    Example:
        {
            'query': 'How do I reset my password?',
            'info': {
                '_info': '**Response is generated by performing semantic search over our internal knowledge base**\n\n**Additional Params:**\n* num_similar_docs: 7\n\n**Referenced documents for this response include::**\n- https://.atlassian.net/wiki/spaces/DOCS/pages/12345\n- https://.atlassian.net/browse/TICKET-6789\n\n[{\'title\': \'Password Reset Runbook\', \'link\': \'...\'}]',
                'is_preapproved': False,
                'runbook_metadata': [{'title': 'Password Reset Runbook', 'link': '...'}]
            },
            'response': 'You can reset your password by following the steps outlined in the linked document.',
            'code': None,
            'datatable': None
        }

    Raises:
        Exception: If an error occurs during the execution of the database query.
    """
    print("===============================KNOWLEDGE BASE SEARCH===============================")
    
    db = pgdbWrapper()
    
    result = RAGInput(query=query, response="", code=None, datatable=None, info={"is_preapproved": False, "_info": None, "runbook": None})
    
    num_similar_docs = int(config.get("configurable", {}).get("num_similar_docs", "7"))
    
    result.info["_info"] = textwrap.dedent(f"""
        **Response is generated by performing semantic search over our internal knowledge base**\n
        **Additional Params:**
        * num_similar_docs: {num_similar_docs}
    """)
       
    try:
        query_embedding = helper.get_embedding_model().embed_query(query)
            
        sql_query = """
            WITH RankedEmbeddings AS (
                SELECT chunk_text, source_id, source
                FROM embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            )
            SELECT
                re.chunk_text,
                CASE
                    WHEN re.source = 1 THEN 'https://.atlassian.net/wiki/spaces/' || cp.space || '/pages/' || cp.pageid
                    WHEN re.source = 2 THEN 'https://.atlassian.net/browse/' || ji.issuekey
                    ELSE NULL
                END AS document_link
            FROM RankedEmbeddings re
            LEFT JOIN confluence_pages cp ON re.source_id = cp.id AND re.source = 1
            LEFT JOIN jira_issues ji ON re.source_id = ji.id AND re.source = 2
        """
        params = (query_embedding, num_similar_docs)
        results = db.execute_select_query(sql_query, params)

        if results: 
            retrieved_documents = [(row, row) for row in results]
            context = "\n".join([doc[0][0] for doc in retrieved_documents])
            document_links = "\n".join([f"- {doc[0][1]}" for doc in retrieved_documents])
            llm_response = generate_llm_response(query, context)
            if isinstance(llm_response, str): llm_response = Utilities.extract_json_from_string(llm_response)
            if isinstance(llm_response, dict) and "answer" in llm_response: 
                result.response = llm_response["answer"]
                result.info["runbook"] = llm_response["runbook_metadata"]
            else: result.response = llm_response
            result.info["_info"] = "\n**Referenced documents for this response include::**\n" + document_links + "\n\n" + result.info["_info"] + "\n"
            # print("================================>>", result)   
    except Exception as e:
        print(f"Error executing query: {e}")
        
    finally:
        db.disconnect()

    return result.model_dump()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
@tool(
    "Search_PreApproved_Responses",
    description=f"""
    **Key Feature:**
    This tool searches *exclusively* within a database of pre-approved responses that have been validated by the user for accuracy. It provides concise, reliable answers.

    **When to Use:**
    -   Use this tool when the user's query requires a precise, factually correct answer.
    -   Use this tool when the user needs a trusted, pre-validated response.
    -   Use this tool when a concise answer is preferred over detailed explanations.

    **When NOT to Use:**
    -   Do NOT use this tool for open-ended questions requiring exploration of a topic.
    -   Do NOT use this tool if the user needs information from raw documentation.
    -   Do NOT use this tool if the query requires combining information from multiple sources.

    **Input Parameters:**
    -   `query` (string): The user's natural language query. This parameter is *mandatory*.

    **Output:**
    -   The most semantically similar pre-approved response, if one is found above the similarity threshold.
    """,
    args_schema= RAGInput,
    return_direct= False    
)
def preapproved_search_tool(query: str, config:RunnableConfig) -> dict:
    """
    Performs a weighted semantic search against a table of pre-approved responses.

    This function takes a user's query, converts it into an embedding, and then performs a semantic search against a PostgreSQL table (`approved_responses`).
    The search considers both the user's query and the previously approved AI responses, using configurable weights to determine their relative importance.

    The function filters the table to include only responses with a rating of 3 or higher and created within the last 30 days. It then calculates a combined
    similarity score based on the weighted distances between the query embedding and the stored embeddings. The function returns the most semantically similar
    response that falls within the specified similarity threshold.

    Args:
        query: The user's input query or prompt (natural language text).

    Returns:
        dict: A dictionary containing:
            -   'query' (str): The original user input.
            -   'info' (dict): A description of the tool's operation, including configuration parameters.
            -   'response' (str): The generated natural language response from the LLM, based on the retrieved preapproved responses. This will be None if an error occurs during the database query.
            -   'code' (str): The code used to generate the response, if applicable.
            -   'datatable' (str): The data table used to generate the response, if applicable.

        Example: {
            'query': 'What is the capital of France?', 
            'info': {
                '_info': 'Response is generated by performing semantic search over our internal knowledge base',
                'is_preapproved': True
            }, 
            'response': 'Paris is the capital of France.', 
            'code': None, 
            'datatable': None
        }
        
    Raises:
        Exception: If an error occurs during the database query execution.
    """
    print("===============================PREAPPROVED RESPONSES===============================")
    
    db = pgdbWrapper()
    
    result = RAGInput(query=query, response="", code=None, datatable=None, info={"is_preapproved": True, "_info": None, "runbook": True})

    similarity_threshold = float(config.get("configurable", {}).get("similarity_threshold", "0.7"))
    query_weight = float(config.get("configurable", {}).get("query_weight", "0.7"))
    feedback_weight = round(float(1 - query_weight), 1)
    
    result.info["_info"] = textwrap.dedent(f"""
        **Response is generated by performing weighted semantic search against a pre-approved responses.**\n
        **Additional Params:**
        * similarity_threshold: {similarity_threshold}
        * query_weight: {query_weight}
        * feedback_weight: {feedback_weight}
    """)
    
    try:
        query_embedding = helper.get_embedding_model().embed_query(query)
        
        sql_query = """
            WITH SimilarityScores AS (
                SELECT response,
                    (
                        %s * (user_query_embedding <-> %s::vector) + 
                        %s * (response_embedding <-> %s::vector)
                    ) AS combined_similarity
                FROM approved_responses
                WHERE NULLIF(rating, '')::int >= 3
                AND created >= NOW() - INTERVAL '30 days'
            )
            SELECT response, combined_similarity
            FROM SimilarityScores
            WHERE combined_similarity < %s
            ORDER BY combined_similarity DESC
            LIMIT 1;
        """
        
        params = (query_weight, query_embedding, feedback_weight, query_embedding, similarity_threshold)
        results = db.execute_select_query(sql_query, params)

        if results: 
            retrieved_documents = [(row) for row in results]
            result.response = "\n".join([doc[0] for doc in retrieved_documents])
            
    except Exception as e:
        print(f"Error executing query: {e}")
        
    finally:
        db.disconnect()
        
    return result.model_dump()
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
@tool(
    "Combined_Hybrid_Search",
    description=f"""
    **Key Feature:**
    This tool combines information from both pre-approved responses and knowledge base documentation to provide a comprehensive answer. It prioritizes validated responses and augments them with relevant knowledge base details.

    **When to Use:**
    -   **IMPORTANT: Use this tool if you are unsure which of the other tools to use.**
    -   Use this tool when the user's query requires a detailed and reliable answer that benefits from both validated information and in-depth knowledge base context.
    -   Use this tool for complex or nuanced questions that may not be fully answered by either source alone.

    **When NOT to Use:**
    -   Do NOT use this tool if the user *only* wants a validated answer (use Search_PreApproved_Responses).
    -   Do NOT use this tool if the user *only* needs information from the raw knowledge base (use Search_Knowledge_Base).

    **Key Difference:**
    -   This tool *synthesizes* information, providing a more complete answer than either of the other tools.

    **Input Parameters:**
    -   `query` (string): The user's natural language query. This parameter is *mandatory*.

    **Output:**
    -   A natural language response generated from the combined information of pre-approved responses and knowledge base documents.
    """,
    args_schema= RAGInput,
    return_direct= False  
)
def hybrid_search_tool(query: str, config:RunnableConfig) -> dict:
    """
    Performs a hybrid semantic search combining pre-approved responses and knowledge base documentation.

    This function strategically retrieves information by first searching for highly relevant and recent
    pre-approved responses based on semantic similarity to the user's query. It then supplements these
    findings with relevant document chunks from the broader knowledge base to ensure a comprehensive
    and well-contextualized response generated by a Large Language Model (LLM).

    The hybrid approach prioritizes validated and curated information while leveraging the breadth
    of the knowledge base for more nuanced or less common queries.

    Args:
        query (str): The user's natural language question or prompt.
        config (RunnableConfig): A configuration object that can contain parameters influencing the search,
                                 such as:
                                 - 'similarity_threshold' (float): The threshold for semantic similarity when
                                   searching pre-approved responses.
                                 - 'query_weight' (float): The weight assigned to the user query embedding
                                   in the pre-approved response similarity calculation.
                                 - 'num_similar_docs' (int): The maximum number of relevant document chunks
                                   to retrieve from the knowledge base.

    Returns:
        dict: A dictionary containing the results of the hybrid search and LLM generation:
            - 'query' (str): The original user input.
            - 'info' (dict): A dictionary providing details about the tool's execution:
                - '_info' (str): A human-readable description of the hybrid search process and the
                  configuration parameters used, including the number of retrieved documents and links.
                - 'is_preapproved' (bool): Indicates if the response is based on pre-approved content
                  (will be True if a high-confidence pre-approved response is primarily used).
                - 'runbook_metadata' (list[dict] or None): If the LLM response includes structured
                  runbook metadata, it will be present here as a list of dictionaries. Otherwise, it will be None.
            - 'response' (str): The natural language answer generated by the LLM, potentially drawing
              from both pre-approved responses and knowledge base documents. If the LLM response
              is a JSON string with an 'answer' key, that value will be used. Otherwise, the raw
              LLM output is provided.
            - 'code' (str or None): Reserved for code snippets generated by the LLM (currently None).
            - 'datatable' (str or None): Reserved for data tables generated by the LLM (currently None).

    Example:
        {
            'query': 'What are the steps to troubleshoot a slow application?',
            'info': {
                '_info': '**Response is generated by performing hybrid semantic search combining pre-approved responses and knowledge base documentation to retrieve relevant information.**\n\n**Additional Params:**\n* num_similar_docs: 7\n* similarity_threshold: 0.7\n* query_weight: 0.7\n* feedback_weight: 0.3\n\n**Referenced documents for this response include::**\n- https://.atlassian.net/wiki/spaces/TECH/pages/9876\n- - [Pre-approved Response: Check network connectivity and server load.]\n\n[{\'title\': \'Application Troubleshooting Runbook\', \'link\': \'...\'}]',
                'is_preapproved': False,
                'runbook_metadata': [{'title': 'Application Troubleshooting Runbook', 'link': '...'}]
            },
            'response': 'To troubleshoot a slow application, first check your network connectivity and the server load. Further details can be found in the linked documentation.',
            'code': None,
            'datatable': None
        }

    Raises:
        Exception: If an error occurs during the execution of any database query.
    """
    print("===============================HYBRID SEARCH===============================")
    
    db = pgdbWrapper()
    
    result = RAGInput(query=query, response="", code=None, datatable=None, info={"is_preapproved": False, "_info": None, "runbook": None})
    
    similarity_threshold = float(config.get("configurable", {}).get("similarity_threshold", "0.7"))
    query_weight = float(config.get("configurable", {}).get("query_weight", "0.7"))
    num_similar_docs = int(config.get("configurable", {}).get("num_similar_docs", "7"))
    feedback_weight = round(float(1 - query_weight), 1)
    
    result.info["_info"] = textwrap.dedent(f"""
        **Response is generated by performing hybrid semantic search combining pre-approved responses and knowledge base documentation to retrieve relevant information.**\n
        **Additional Params:**
        * num_similar_docs: {num_similar_docs}
        * similarity_threshold: {similarity_threshold}
        * query_weight: {query_weight}
        * feedback_weight: {feedback_weight}
    """)
    
    try:
        query_embedding = helper.get_embedding_model().embed_query(query)
        
        sql_query = """
            WITH approved AS (
                SELECT ar.response
                FROM approved_responses ar
                WHERE (
                    %s * (user_query_embedding <-> %s::vector) +
                    %s * (response_embedding <-> %s::vector)
                ) < %s
                AND NULLIF(rating, '')::int >= 3
                AND created >= NOW() - INTERVAL '30 days'
                ORDER BY rating desc, created desc
                LIMIT 3
            ),
            embdg_base AS (
                SELECT chunk_text, source_id, source
                FROM embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT (10 - (SELECT COUNT(1) FROM approved))
            ),
            embdg_with_link AS (
                SELECT
                    eb.chunk_text,
                    CASE
                        WHEN eb.source = 1 THEN 'https://.atlassian.net/wiki/spaces/' || cp.space || '/pages/' || cp.pageid
                        WHEN eb.source = 2 THEN 'https://.atlassian.net/browse/' || ji.issuekey
                        ELSE NULL
                    END AS document_link
                FROM embdg_base eb
                LEFT JOIN confluence_pages cp ON eb.source_id = cp.id AND eb.source = 1
                LEFT JOIN jira_issues ji ON eb.source_id = ji.id AND eb.source = 2
            )
            SELECT response, 'NA' AS document_link FROM approved
            UNION ALL
            SELECT chunk_text, document_link FROM embdg_with_link
            LIMIT %s;
        """
        params = (query_weight, query_embedding, feedback_weight, query_embedding, similarity_threshold, query_embedding, num_similar_docs)
        results = db.execute_select_query(sql_query, params)

        if results: 
            retrieved_documents = [(row, row) for row in results]
            context = "\n".join([doc[0][0] for doc in retrieved_documents])
            document_links = "\n".join([f"- {doc[0][1]}" for doc in retrieved_documents if doc[0][1] != 'NA'])
            llm_response = generate_llm_response(query, context)
            if isinstance(llm_response, str): llm_response = Utilities.extract_json_from_string(llm_response)
            if isinstance(llm_response, dict) and "answer" in llm_response: 
                result.response = llm_response["answer"]
                result.info["runbook"] = llm_response["runbook_metadata"]
            else: result.response = llm_response
            result.info["_info"] = "\n**Referenced documents for this response include::**\n" + document_links + "\n\n" + result.info["_info"] + "\n"
               
    except Exception as e:
        print(f"Error executing query: {e}")
        
    finally:
        db.disconnect()
        
    return result.model_dump()
#--------------------------------------------------------------------------------------------
