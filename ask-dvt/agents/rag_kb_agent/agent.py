import os, sys, json, textwrap, urllib3, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from state import AgentState
from agents.rag_kb_agent.helper import RAGHelper, RAGInput, RAGOutput
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from agents.rag_kb_agent.tools import kb_search_tool, preapproved_search_tool, hybrid_search_tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class RAGAgent:
    def __init__(self, thread_id="abc123", debug=False, add_memory_saver: bool = True):
        self.helper = RAGHelper()
        self.tools = [kb_search_tool, preapproved_search_tool, hybrid_search_tool]
        self.llm = self.helper.get_llm().bind_tools(self.tools)
        self.memory_saver = MemorySaver() if add_memory_saver else None
        self.tool_input_schema = json.dumps(RAGInput.model_json_schema())
        self.tool_output_schema = json.dumps(RAGOutput.model_json_schema())
        self.custom_prompt = textwrap.dedent(f"""
            **Tool Input Schema**
            When invoking the tools, you must **strictly** adhere to the following JSON schema:
            {self.tool_input_schema}
            
            **Tool Output Schema**
            The tools will return responses in the following JSON schema:
            {self.tool_output_schema}
        """)
        self.agent_prompt = SystemMessage(content=self.helper.get_agent_prompt() + "\n" + self.custom_prompt)
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt,
            checkpointer=self.memory_saver,
            name="rag_kb_agent",
            state_schema=AgentState,
            debug=debug
        )
        
    def get_agent(self):
        return self.agent
    
    def run_agent(self, user_input, config):
        def extract_last_tool_message(result) -> ToolMessage | None:
            for msg in reversed(result["messages"]):
                if isinstance(msg, ToolMessage):
                    return msg
            return None
        
        def extract_last_ai_message(result) -> AIMessage | None:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    return msg
            return None
        
        original_question = user_input
        current_question = original_question
        
        for attempt in range(3):
            result = self.agent.invoke({"messages": [HumanMessage(content=current_question)]}, config)
            
            tool_call_id, tool_msg = None, extract_last_tool_message(result)
            ai_msg_id, ai_msg = None, extract_last_ai_message(result)
            
            if tool_msg and hasattr(tool_msg, 'tool_call_id'):
                tool_call_id = tool_msg.tool_call_id
            
            if ai_msg and hasattr(ai_msg, 'response_metadata') and "STOP" in ai_msg.response_metadata["finish_reason"]:
                if hasattr(ai_msg, 'tool_calls') and len(ai_msg.tool_calls) > 0: ai_msg_id = ai_msg.tool_calls[0]["id"]
                else: return result
                
            if tool_call_id == ai_msg_id:
                return result
            
            time.sleep(5)
            if attempt == 0: current_question = f"{original_question}. Please use a tool to find the answer."
            elif attempt == 1: current_question =  f"You MUST use the available tools to answer this question: {original_question}"
            else: current_question =  f"It is critical that you use a tool to answer this: {original_question}. Do not provide an answer without using a tool."
        
        return {"messages": [AIMessage(content="I could not find the right tool to fulfill your request. Please rephrase or provide more context.")]}

    def rag_agent_node(self, state: AgentState, config: RunnableConfig) -> dict:
        # print(config["configurable"].get("r", 1.0))
        result = self.agent.invoke({"messages": state["messages"]}, config)
        return {"messages": result["messages"]}
                
# rag_agent = RAGAgent()
# result = rag_agent.run_agent("Pipeline Dependency not working as expected for Add/Delete and Triggers process. Knowledge Base responses only")
# print(result["messages"][-1].content)
# for msg in result["messages"]:
#     if isinstance(msg, AIMessage):
#         print("====================================================================================")
#         print(msg.tool_calls)
#         print("====================================================================================")