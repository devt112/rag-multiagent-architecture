import streamlit as st
from pydantic import Field
import os, sys, json, textwrap, urllib3, time, yaml, logging, uuid
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from state import AgentState, ConfigSchema
from typing import Optional, List, Dict, Any
from commons.utils import Utilities
from commons.logger import CustomLogger
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class RunBookState(AgentState):
    actions: Optional[List[Dict[str, Any]]] = Field([], title="Actions", description="List of actions to be executed")

class RunbookAgent:
    def __init__(self):
        my_logger = CustomLogger(
            level=logging.DEBUG,
            log_format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
        self.logger = my_logger.get_logger()
        
        workflow = StateGraph(RunBookState, config_schema=ConfigSchema)
        workflow.add_node("initialize", self._initialize_)
        workflow.add_node("get_runbook_details", self._get_runbook_details_)
        workflow.add_node("execute_action", self._execute_action_)
        workflow.set_entry_point("initialize")
        workflow.add_conditional_edges(
            "initialize", 
            lambda state: state["next_node"], 
            ["get_runbook_details", END]
        )
        workflow.add_conditional_edges(
            "get_runbook_details", 
            lambda state: state["next_node"], 
            ["execute_action", END]
        )
        self.graph = workflow.compile(name="runbook_agent")
        
    def _initialize_(self, state:RunBookState, config:RunnableConfig):
        """
        Presents the Agent's output and relevant context to a human for review and approval,
        potentially triggering an interrupt to pause the LangGraph state graph.

        This node in the LangGraph state graph extracts the latest tool messages
        from the agent's state. If a 'runbook' is present within the extracted
        information, it triggers an 'interrupt' mechanism. This pause allows
        for human inspection and feedback on the proposed action before proceeding.
        The workflow will either transition to 'get_runbook_details' upon human
        approval ("Yes" in the feedback) or terminate at the 'END' node otherwise.

        Args:
            state (RunBookState): The current state of the agent within the LangGraph,
                containing the message history, LLM output, and any relevant context.

        Returns:
            dict: A dictionary indicating the next node in the state graph.
                Returns `{"next_node": "get_runbook_details"}` if the human approves
                the action, and `{"next_node": END}` if the action is not approved.
                The 'interrupt' itself does not return a value in the traditional
                sense but rather pauses the graph execution pending external input.
        """
        print("===============================HUMAN IN THE LOOP===============================")

        is_approved = None
        _agent_response_dict = Utilities.extract_tool_messages(state["messages"])
        if _agent_response_dict and _agent_response_dict.get("info", None) and _agent_response_dict["info"].get("runbook", None) and isinstance(_agent_response_dict["info"]["runbook"], list) and len(_agent_response_dict["info"]["runbook"]) > 0:
            selected_action = interrupt({
                "action_prompt": "Automated Action Available",
                "action_context": "An operational runbook has been identified that can automatically perform the steps needed for your request, potentially providing a quicker resolution. Would you like to proceed with the automated execution now?",
                "action_controls": [{"actions": ["Yes", "No"]}]
            })
            self.logger.info("User Query: %s | POST FIRST INTERRUPT | %s", _agent_response_dict["query"], selected_action.get("sel_action", None))
            if selected_action.get("sel_action", None) and selected_action["sel_action"] == "Yes": 
                return {"next_node": "get_runbook_details"}
            else: return {"next_node": END}
        else: return {"next_node": END}
    
    def _get_runbook_details_(self, state:RunBookState, config:RunnableConfig):
        _agent_response_dict = Utilities.extract_tool_messages(state["messages"])
        
        if _agent_response_dict and _agent_response_dict.get("info", None) and _agent_response_dict["info"].get("runbook", None) and isinstance(_agent_response_dict["info"]["runbook"], list) and len(_agent_response_dict["info"]["runbook"]) > 0:
            runbook =_agent_response_dict["info"]["runbook"][0]        
            if runbook.get("has_runbook", False):
                runbook_name = runbook.get("runbook_name", None)
                runbook_yml = RunbookAgent.get_runbook(runbook_name)
                
                _actions_, _actions_list_ = RunbookAgent.read_actions_from_yaml(runbook_yml)
                selected_action = interrupt({
                    "action_prompt": "Select an Action to Execute",
                    "action_context": "An operational runbook relevant to your request contains the following automated actions. To proceed, please select the action corresponding to the action you wish to execute",
                    "action_controls": [{"actions": _actions_list_}]
                })

                _action_ = next((action for action in _actions_ if action['name'] ==  selected_action.get("sel_action", None)), None)
                self.logger.info("User Query: %s | POST SECOND INTERRUPT | %s | %s", _agent_response_dict["query"], selected_action.get("sel_action", None), runbook_name)
              
                if selected_action.get("sel_action", None) and _action_:
                    _action_parameters_ = _action_.get("parameters", [])
                    
                    if _action_parameters_ and isinstance(_action_parameters_, list) and len(_action_parameters_) > 0:
                        text_parameters = [{
                            "name": p.get("name"), 
                            "display_name": p.get("display_name"), 
                            "description": p.get("description"), 
                            "secret": p.get("secret", False)
                        } for p in _action_parameters_ if p.get('type') == 'text']

                        choice_parameters = [{
                            "name": p.get("name"),
                            "display_name": p.get("display_name"), 
                            "description": p.get("description"),
                            "options": p.get("options", [])
                        } for p in _action_parameters_ if p.get("type") == "choice"]

                        submitted_values = interrupt({
                            "action_prompt": "Please provide the required parameters for the selected action",
                            "action_context": "The selected action requires the following parameters to proceed. Please provide the necessary values for each parameter.",
                            "action_controls": [{
                                "text_inputs": text_parameters,
                                "choice_inputs": choice_parameters
                            }]
                        })
                    
                        self.logger.info("User Query: %s | POST THIRD INTERRUPT | %s", _agent_response_dict["query"], submitted_values)
                        
                        if submitted_values:
                            return {
                                "next_node": "execute_action",
                                "actions": [{
                                    "_action_": _action_,
                                    "_action_params": submitted_values
                                }]
                            }
        return {"next_node": END}
    
    def _execute_action_(self, state:RunBookState, config:RunnableConfig):
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(state["actions"])
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        executor_result = {"executor_result": {
            "status":"Agent has executed the action successfully",
            "actions": state["actions"]
        }}
        new_message = ToolMessage(
            content=json.dumps(executor_result),
            tool_call_id=str(uuid.uuid4()),
            name="runbook_executor"
        )
        return {"messages": state["messages"] + [new_message]}
    
    def get_agent(self):
        return self.graph
    
    @staticmethod
    def read_actions_from_yaml(yaml_data):
        try:
            data = yaml.safe_load(yaml_data)
            actions = data.get('runbook', {}).get('actions', [])
            action_names = [action['name'] for action in actions if 'name' in action]
            return actions, action_names
        except yaml.YAMLError as e:
            print(f"Error reading YAML: {e}")
            return []
        
    @staticmethod
    @st.cache_data
    def get_runbook(runbook_name):
        runbook_yml = Utilities.read_file_from_gcs("GCP Project", f"Devesh/runbooks/{runbook_name}")
        return runbook_yml
        # with open(r"C:\dev\99999_DVT_ASKGBP_AI_MULTIAGENT\ask-dvt\agents\hil_runbook\sample_runbook.yml", 'r') as file:
        #     return file.read()