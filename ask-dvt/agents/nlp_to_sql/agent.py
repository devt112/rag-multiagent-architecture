import os, sys, json, textwrap, urllib3, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from state import AgentState
from typing import List, Dict, Any
from agents.nlp_to_sql.helper import NLPToSQLHelper, NLPToSQLToolInput
from configuration import Configuration
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from agents.nlp_to_sql.tools import generate_execute_sql_query, generate_data_insights
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class NLPSQLAgent:
    """
    A class for handling Natural Language to SQL queries using a language agent.
    """

    def __init__(self, thread_id: str = "default_thread", debug: bool = False, add_memory_saver: bool = True):
        """
        Initializes the NLPSQLAgent.

        Args:
            thread_id (str, optional): The ID of the thread. Defaults to "default_thread".
            debug (bool, optional):  A flag to enable or disable debugging. Defaults to False.
        """
        self.helper = NLPToSQLHelper()
        self.memory_saver = MemorySaver() if add_memory_saver else None
        self.tools = [generate_execute_sql_query, generate_data_insights]
        self.llm = self.helper.get_llm().bind_tools(self.tools)
        self.tool_input_schema = json.dumps(NLPToSQLToolInput.model_json_schema())
        self.base_prompt = self.helper.get_agent_prompt()
        self.custom_prompt = textwrap.dedent(f"""
            **Tool Input Schema**
            When invoking the tools, you must **strictly** adhere to the following JSON schema:
            {self.tool_input_schema}
            
            **Tool Output Schema**
            The tools will return responses in the following JSON schema:
            {self.tool_input_schema}
        """)
        self.agent_prompt = SystemMessage(content=self.base_prompt)
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt,
            checkpointer=self.memory_saver,
            name="nlp-to-sql",
            state_schema=AgentState,
            debug=debug,
        )
        self.config = {"configurable": {"thread_id": thread_id}}

    def run_agent(self, user_input):
        return self.agent.invoke({"messages": [HumanMessage(content=user_input)]}, self.config)

    def get_agent(self):
        return self.agent

    def nlpsql_agent_node(self, state: AgentState, config: RunnableConfig) -> dict:
        # print(config["configurable"].get("r", 1.0))
        result = self.agent.invoke({"messages": state["messages"]}, config)
        return {"messages": result["messages"]}
