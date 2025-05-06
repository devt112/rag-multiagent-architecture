import os, sys, warnings, time, json, markdown, re, urllib3
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import concurrent.futures
from state import ToolBaseSchema
from agents.supervisor.agent import SuperVisor
from commons.feedback_manager import FeedbackManager
from commons.utils import Utilities
import pandas as pd
from io import StringIO


class Processor:
    def __init__(self):
        self.supervisor = SuperVisor()
        self.config = {
            "configurable":{
                "thread_id": "46252", 
                "num_similar_docs": "8", 
                "similarity_threshold": "0.7", 
                "query_weight": "0.7"
            }
        }
        self._agent_response_dict = None
        self._agent_response_obj = None
        self.feedback_manager = FeedbackManager()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    def ensure_markdown_formatting(self, text):
        starts_with_markdown = text.strip().startswith("```markdown")
        ends_with_backticks = text.strip().endswith("```")

        if not starts_with_markdown or not ends_with_backticks:
            return "```markdown\n" + text + "\n```"
        else:
            return text
    
    def set_agent_config(self, user_config):
        self.config = user_config

    def process_user_query(self, user_query, user_config=None, chat_history=None):  # Add chat_history
        _agent_response_dict, response = None, None
        config_to_use = user_config if user_config is not None else self.config

        try:
            _agent_result, _agent_state = self.supervisor.invoke(user_query, config_to_use, chat_history)
            _agent_response_dict = Utilities.extract_tool_messages(_agent_result["messages"])
            if _agent_response_dict and not isinstance(_agent_response_dict, dict) and isinstance(_agent_response_dict, str):
                _agent_response_dict = Utilities.extract_json_from_string(_agent_response_dict)

            self._agent_response_obj = ToolBaseSchema(**_agent_response_dict) if _agent_response_dict else None
            return self._agent_response_obj, _agent_state
        except Exception as e:
            print(f"Error executing query: {e}")

        return None
    
    def get_agent_response_obj(self):
        return self._agent_response_obj
    
    def save_feedback(self, user_query, _agent_response_obj, rating, feedback_text):
        response, code = None, Processor._get_key_from_obj(_agent_response_obj, "code")
        response = code if code else Processor._get_key_from_obj(_agent_response_obj, "response")
        feedback_data = {
            "response": response,
            "rating": str(rating),
            "user_feedback": feedback_text,
            "user_query": user_query,
            "embeddings": {
                "response": "response_embedding", 
                "user_query": "user_query_embedding"
            }
        }
        future = self.executor.submit(self.feedback_manager.save_feedback(feedback_data, table_name="approved_responses"))
        return future

    @staticmethod
    def _get_key_from_obj(_agent_response_obj, key):
        if _agent_response_obj and key and hasattr(_agent_response_obj, key):
            return getattr(_agent_response_obj, key)
        elif not key:
            return _agent_response_obj
        else: return None
    
    @staticmethod
    def _to_obj(_agent_response_dict):
        return ToolBaseSchema(**_agent_response_dict) if _agent_response_dict else None
        