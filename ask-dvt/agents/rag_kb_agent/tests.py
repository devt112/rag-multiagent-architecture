import time
from tools import kb_search_tool, preapproved_search_tool, hybrid_search_tool
from agent import RAGAgent
from typing import List, Dict, Any
from agents.rag_kb_agent.helper import RAGHelper
from langchain_core.messages import AIMessage

helper = RAGHelper()

def extract_last_ai_message(result) -> AIMessage | None:
    """
    Extracts the last AIMessage from the result.
    """
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None

def test_kb_search(query: str, tool_config: dict):
    return kb_search_tool.invoke({"user_input": query, "tool_config": tool_config}, config={"configurable": tool_config})

def test_preapproved_search(query: str, tool_config: dict):
    return preapproved_search_tool.invoke({"user_input": query, "tool_config": tool_config}, config={"configurable": tool_config})

def test_hybrid_search(query: str, tool_config: dict):
    return hybrid_search_tool.invoke({"user_input": query, "tool_config": tool_config}, config={"configurable": tool_config})

def test_agent(user_inputs: List[str]):
    rag_agent = RAGAgent(debug=False, add_memory_saver=False)
    sources = [
        {"src": "Hybrid Responses", "tool": "Hybrid_Combined_Responses"},
        {"src": "Knowledge Base", "tool": "Search_Knowledge_Base"},
        {"src": "Preapproved Responses", "tool": "Search_PreApproved_Responses"}
    ]
    for source in sources:
        for user_input in user_inputs:
            user_query = user_input.strip() + " Search " + source["src"] + " only."
            print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            print(user_query)
            print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            result = rag_agent.run_agent(user_query, {"configurable": {
                "thread_id": "46252", 
                "num_similar_docs": "8", 
                "similarity_threshold": "0.7", 
                "query_weight": "0.6"
            }})
            print(len(result["messages"]), "\n----------------------------------------------------------------------------------------------\n", result["messages"][-1].content)
            time.sleep(5)
            print("----------------------------------------------------------------------------------------------\n\n")
            
def test_graph():
    pass

if __name__ == "__main__":
    # user_inputs = [
    #     "Pipeline Dependency not working as expected for Add/Delete and Triggers process.",
    #     "How to handle repeated events in dvt to avoid congestion?.",
    #     "dvt sends the same event repeatedly. It is congesting dvt listener since there are too many events.",
    #     "Dag is not getting published because of subdag already exists."
    # ]   
    # tool_config = {'num_similar_docs': 9, 'similarity_threshold': 0.5, 'query_weight': 0.5}
    # print(test_preapproved_search(query, tool_config)["response"])
    # test_agent(user_inputs)

    query_embedding = helper.get_embedding_model().embed_query(query)
    with open(r"C:\dev\dump\embedding.txt", "w") as file:
        file.write(str(query_embedding))
