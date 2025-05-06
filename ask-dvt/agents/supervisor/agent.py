import os, sys, json, textwrap, urllib3, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from configuration import Configuration
from state import AgentState, ConfigSchema
from langgraph.graph import END, START, StateGraph
from agents.supervisor.supervisor import create_supervisor
from agents.hil_runbook.graph import RunbookAgent
from agents.rag_kb_agent.agent import RAGAgent
from agents.nlp_to_sql.agent import NLPSQLAgent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage, HumanMessage

class SuperVisor:
    def __init__(self):
        self.memory = InMemorySaver()
        self.store = InMemoryStore()
        self.rag_agent = RAGAgent().get_agent()
        self.nlpsql_agent = NLPSQLAgent().get_agent()
        self.runbook_agent = RunbookAgent().get_agent()
        self.supervisor_workflow = create_supervisor(
            [self.rag_agent, self.nlpsql_agent],
            supervisor_name="ask_gbp_supervisor",
            model=Configuration.query_model,
            prompt=Configuration.supervisor_agent_prompt,
            state_schema=AgentState,
            output_mode="full_history",
            add_handoff_back_messages=False,
            debug=False
        )
        self.supervisor_app = self.supervisor_workflow.compile(checkpointer=self.memory, store=self.store, name="ask_gbp_supervisor")
        self.add_human_in_the_loop()
        
    def add_human_in_the_loop(self):
        builder = StateGraph(AgentState)
        builder.add_node(self.supervisor_app)
        builder.add_node(self.runbook_agent)
        builder.add_edge(START, self.supervisor_app.name)
        builder.add_edge(self.supervisor_app.name, self.runbook_agent.name)
        self.supervisor_app = builder.compile(checkpointer=self.memory, store=self.store, name="ask_gbp_supervisor_hil")
        
    def get_supervisor_agent(self, hil=False):
        return self.compile_workflow(hil)
    
    def invoke(self, user_input, config, chat_history=None):
        if type(user_input) == str:
            messages = []
            if chat_history:
                messages.extend([HumanMessage(content=message) if "User:" in message else AIMessage(content=message) for message in chat_history])
            # print("======================================================================")
            # print(chat_history)
            # print("======================================================================")
            messages.append(HumanMessage(content=user_input))
            result = self.supervisor_app.invoke({"messages": messages}, config)
        else:
            result = self.supervisor_app.invoke(user_input, config)

        return result, self.supervisor_app.get_state(config=config)

    def stream(self, user_input, config, chat_history=None):
        inputs = {"messages": [HumanMessage(content=user_input)]} if type(user_input) == str else user_input
        for event in self.supervisor_app.stream(inputs, config, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_call_message = self.extract_tool_messages(message)
            else:
                message.pretty_print()
                if isinstance(message, AIMessage):
                    output_channel.write(message.content)