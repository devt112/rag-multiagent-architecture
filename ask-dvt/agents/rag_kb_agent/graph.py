import os, sys, json, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from agent import RAGAgent
from state import AgentState, ConfigSchema
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage

class rag_agent_graph:
    def __init__(self, debug=False, use_memory=False):
        memory = MemorySaver() if use_memory else None
        workflow = StateGraph(AgentState, config_schema=ConfigSchema)
        workflow.add_node("agent", RAGAgent(debug=debug).rag_agent_node)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        # graph = workflow.compile()
        self.graph = workflow.compile(checkpointer=memory) # Set to None for testing purposes, otherwise use memory

    def get_examples(self):
        user_inputs = [
            "Pipeline Dependency not working as expected for Add/Delete and Triggers process.",
            "How to handle repeated events in dvt to avoid congestion?.",
            "dvt sends the same event repeatedly. It is congesting dvt listener since there are too many events.",
            "Dag is not getting published because of subdag already exists."
        ]

        sources = [
            {"src": "Knowledge Base", "tool": "Search_Knowledge_Base"},
            {"src": "Preapproved Responses", "tool": "Search_PreApproved_Responses"},
            {"src": "Hybrid (Combination of other internal sources) Responses", "tool": "Hybrid_Combined_Responses"}
        ] 
        return user_inputs, sources
 
    def test_graph_single_input(self):
        user_input = "What is Header.yml? Please describe in detail. Search Knowledge Base only."
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        result = self.graph.invoke(initial_state, {"configurable":{
            "thread_id": "46252", 
            "num_similar_docs": "8", 
            "similarity_threshold": "0.7", 
            "query_weight": "0.7"
        }})
        print(len(result["messages"]), "\n----------------------------------------------------------------------------------------------\n", result["messages"][-1].content)

    def test_graph_multiple_inputs(self):
        user_inputs, sources = self.get_examples()
        for user_input in user_inputs:
            for source in sources:
                user_query = user_input.strip() + " Search " + source["src"] + " only."
                
                print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                print(user_query)
                print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                
                initial_state = {"messages": [HumanMessage(content=user_query)]}
                result = self.graph.invoke(initial_state, {"configurable":{
                    "thread_id": "46252", 
                    "num_similar_docs": "8", 
                    "similarity_threshold": "0.7", 
                    "query_weight": "0.7"
                }})
                print(result["messages"][-1].content)
                time.sleep(5)
                print("----------------------------------------------------------------------------------------------\n\n")

    def _invoke_(self, initial_state, configurable):
        result = self.graph.invoke(initial_state, {"configurable": configurable})
        return result

if __name__ == "__main__":
    node = rag_agent_graph()
    node.test_graph_multiple_inputs()