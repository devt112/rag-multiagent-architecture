import os, sys, re, json, urllib3
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from db_helper import pgdbWrapper
from agents.rag_kb_agent.helper import RAGHelper, RAGInput
from typing_extensions import Annotated
from langchain.tools import StructuredTool, tool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

helper = RAGHelper()

def generate_llm_response(query, context):
    prompt = f"""You are a helpful and informative assistant.  Your task is to answer questions based on the provided context.

        Instructions:
        1. **Answer in Detail:** Provide a comprehensive answer to the question, explaining the reasoning behind your response.  Don't just give a one-sentence answer.
        2. **Code Sample (If Applicable):** If the question requires or would benefit from a code sample, provide a relevant example in a suitable programming language. Choose the language that best fits the context of the question or the user's needs. If no code sample is needed, skip this section.
        3. **Troubleshooting Steps (If Applicable):** If the question relates to a technical issue, suggest potential solution or troubleshooting steps.  Be specific and provide clear instructions.  If no troubleshooting is needed, skip this section.
        4. **Confluence/Jira Links (If Applicable):** If you know of any relevant Confluence pages or Jira tickets that could help the user, include the links in your response.  If no links are available, skip this section.
        5. **If No Answer:** If the context does not contain the answer, say "I don't know" or "The answer is not in the provided information." Do NOT hallucinate or make up information.
        6. **Clarity:** Format your response in a clear and easy-to-read manner, using headings, subheadings, bullet points, and numbered lists where appropriate.
        
        Context:
        {context}
        Question:
        {query}
    """
    response = helper.get_llm().invoke(prompt)
    return response.content

#--------------------------------------------------------------------------------------------
@tool(
    "Search_Knowledge_Base",
    description=f"""
    **Key Feature**
    This tool searches for response to user query, exclusively within an internal database of knowledge base documents.

    **When to Use:**
    - For detailed or complex questions requiring up-to-date information from internal documentation.
    - When you need to find specific information within the knowledge base documents.

    **Input Parameters:**
    - `user_input` (string): The user's natural language query.
    Always provide the input parameter (user_input) when invoking the kb_search_tool tool
    
    **Important Notes:**
    - This tool does not include pre-approved responses and searches only the raw knowledge base.
    """,
    args_schema= RAGInput,
    return_direct= False
)
def kb_search_tool(user_input: str, config:RunnableConfig) -> dict:
    """
    Performs semantic search over a internal knowledge base to answer user queries.

    This function retrieves relevant document chunks from a vector database based on the semantic similarity 
    between the user's query and the document content. The retrieved chunks are then concatenated and provided 
    as context to a Large Language Model (LLM), which generates a natural language response.

    Key Characteristics:
    - **Semantic Search:** Employs vector similarity to identify relevant information.
    - **Direct Knowledge Base Query:** Searches a vector database of raw documentation, bypassing pre-approved or validated responses.
    - **LLM Contextualization:** Leverages retrieved document chunks as context for dynamic response generation.

    Args:
        - user_input (str): The user's question or prompt (natural language text).

    Returns:
        dict: A dictionary containing the original user input, tool name, tool configuration, and the generated response.
        Example: {'query': 'What is the capital of France?', 'tool_name': 'Search_Knowledge_Base', 'tool_config': {'num_similar_docs': 7}, 'response': 'Paris'}

    Raises:
        Exception: If an error occurs during the database query execution.
    """
    print("===============================KNOWLEDGE BASE SEARCH===============================")
    
    db = pgdbWrapper()
    result = {
        "query": user_input, 
        "tool_name": "Search_Knowledge_Base", 
        "tool_config": None, 
        "response": None
    }
    
    num_similar_docs = int(config.get("configurable", {}).get("num_similar_docs", "7"))
    result["tool_config"] = {"num_similar_docs": str(num_similar_docs)}
       
    try:
        query_embedding = helper.get_embedding_model().embed_query(user_input)
            
        sql_query = """
            SELECT chunk_text 
            FROM embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """
        params = (query_embedding, num_similar_docs)
        results = db.execute_select_query(sql_query, params)

        if results: 
            retrieved_documents = [(row) for row in results]
            context = "\n".join([doc[0] for doc in retrieved_documents])
            result["response"] = generate_llm_response(user_input, context)
               
    except Exception as e:
        print(f"Error executing query: {e}")
        
    finally:
        db.disconnect()
        
    return result
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
@tool(
    "Search_PreApproved_Responses",
    description=f"""
    **Key Feature**
    - This tool provides searches for responses exclusively within a database of responses that have been rigorously validated and approved for accuracy and usefulness. It operates independently, focusing solely on these pre-approved entries, and does not search knowledge base documentation or combine results from other sources.

    **When to Use: For Trusted, Pre-Validated Answers**
    - Use this tool when the user's query demands answers that are known to be accurate and reliable, drawn specifically from pre-approved or user validated responses. Avoid using it for general information searches or when needing to combine information from multiple sources.

    **Input Parameters:**
    - `user_input` (string): The user's natural language query.
    Always provide the input parameter (user_input) when invoking the preapproved_search_tool tool
    """,
    args_schema= RAGInput,
    return_direct= False    
)
def preapproved_search_tool(user_input: str, config:RunnableConfig) -> dict:
    """
    Performs a weighted semantic search against a table of pre-approved responses.

    This function takes a user's query, converts it into an embedding, and then performs a semantic search against a PostgreSQL table (`approved_responses`).
    The search considers both the user's query and the previously approved AI responses, using configurable weights to determine their relative importance.

    The function filters the table to include only responses with a rating of 3 orf higher and created within the last 30 days. It then calculates a combined
    similarity score based on the weighted distances between the query embedding and the stored embeddings. The function returns the most semantically similar
    response that falls within the specified similarity threshold.

    Args:
        user_input: The user's input query or prompt (natural language text).

    Returns:
        dict: A dictionary containing the original user input, tool name, tool configuration, and the generated response. 
        The generated response is the most semantically similar pre-approved response as a string, or None.
        Example: {'query': 'What is the capital of France?', 'tool_name': 'Search_PreApproved_Responses', 'tool_config': {'similarity_threshold': 0.5, 'query_weight': 0.7}, 'response': 'Paris'}

    Raises:
        Exception: If an error occurs during the database query execution.
    """
    print("===============================PREAPPROVED RESPONSES===============================")
    
    db = pgdbWrapper()
    result = {
        "query": user_input, 
        "tool_name": "Search_PreApproved_Responses", 
        "tool_config": None, 
        "response": None
    }
    
    similarity_threshold = float(config.get("configurable", {}).get("similarity_threshold", "0.7"))
    query_weight = float(config.get("configurable", {}).get("query_weight", "0.7"))
    feedback_weight = round(float(1 - query_weight), 1)
    
    result["tool_config"] = {
        "similarity_threshold": str(similarity_threshold), 
        "query_weight": str(query_weight)
    }
    
    try:
        query_embedding = helper.get_embedding_model().embed_query(user_input)
        
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
            result["response"] = "\n".join([doc[0] for doc in retrieved_documents])
            
    except Exception as e:
        print(f"Error executing query: {e}")
        
    finally:
        db.disconnect()
        
    return result
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
@tool(
    "Combined_Hybrid_Search",
    description=f"""
    This tool combines information from both pre-approved responses and knowledge base documentation to answer user queries.
    **IMPORTANT: Use this tool, if you are unsure which of the other tools to use.**

    **Key Feature: Comprehensive Information Retrieval**
    - This tool provides a comprehensive response by retrieving information from two sources: a curated vector database of pre-approved responses and a vector database of knowledge base documentation. It intelligently combines these sources (prioritizing highly rated, recent pre-approved responses and augments them with relevant document chunks from the knowledge base) to provide a well-rounded and informative answer.

    **When to Use: For Detailed and Reliable Answers**
    - Employ this tool when the user's query requires a detailed and reliable answer that may benefit from a combination of accurate pre-approved responses and additional information from the knowledge base documentation.

    **Key Difference:**
    - This tool is designed to synthesize responses, rather than relying on only one source.

    **Input Parameters:**
    - `user_input` (string): The user's natural language query.
    Always provide the input parameter (user_input) when invoking the hybrid_search_tool tool.
    """,
    args_schema= RAGInput,
    return_direct= False  
)
def hybrid_search_tool(user_input: str, config:RunnableConfig) -> dict:
    """
    This function performs a hybrid semantic search combining pre-approved responses and knowledge base documentation to retrieve relevant information.
    It first searches for semantically similar responses from the pre-approved responses table, prioritizing highly rated and recent entries.
    Then, it supplements these results with relevant document chunks from the knowledge base, ensuring a comprehensive response.
    The retrieved information is concatenated and used as context for a Large Language Model (LLM) to generate a final natural language response.

    Args:
        user_input (str): The user's question or prompt (natural language text).

    Returns:
        dict: A dictionary containing the original user input, tool name, tool configuration, and the generated response. The response is generated from the combined retrieved information and the LLM. Returns None if no result is found.
        Example: {'query': 'What is the capital of France?', 'tool_name': 'Hybrid_Combined_Responses', 'tool_config': {'num_similar_docs': 7, 'similarity_threshold': 0.6, 'query_weight': 0.8}, 'response': 'Paris'}

    Raises:
        Exception: If an error occurs during the database query execution.
    """
    print("===============================HYBRID SEARCH===============================")
    
    db = pgdbWrapper()
    result = {
        "query": user_input, 
        "tool_name": "Hybrid_Combined_Responses", 
        "tool_config": None, 
        "response": None
    }
    
    similarity_threshold = float(config.get("configurable", {}).get("similarity_threshold", "0.7"))
    query_weight = float(config.get("configurable", {}).get("query_weight", "0.7"))
    num_similar_docs = int(config.get("configurable", {}).get("num_similar_docs", "7"))
    feedback_weight = round(float(1 - query_weight), 1)
    
    result["tool_config"] = {
        "num_similar_docs": str(num_similar_docs), 
        "similarity_threshold": str(similarity_threshold), 
        "query_weight": str(query_weight)
    }
    
    try:
        query_embedding = helper.get_embedding_model().embed_query(user_input)
        
        sql_query = """
            with approved as (
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
            embdg as (
                SELECT chunk_text
                FROM embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT (10-(Select count(1) from approved))
            )
            SELECT response FROM approved
            UNION ALL
            SELECT chunk_text FROM embdg
            LIMIT %s;
        """
        params = (query_weight, query_embedding, feedback_weight, query_embedding, similarity_threshold, query_embedding, num_similar_docs)
        results = db.execute_select_query(sql_query, params)

        if results: 
            retrieved_documents = [(row) for row in results]
            context = "\n".join([doc[0] for doc in retrieved_documents])
            result["response"] = generate_llm_response(user_input, context)
            
    except Exception as e:
        print(f"Error executing query: {e}")
        
    finally:
        db.disconnect()
        
    return result
#--------------------------------------------------------------------------------------------
