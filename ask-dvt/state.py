from langgraph.graph import MessagesState
from pydantic import Field
from typing import Optional, Dict, Any
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field, ValidationError
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.graph.message import AnyMessage, add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: RemainingSteps

# class AgentState(MessagesState):
#     config: Optional[Dict[str, Any]] = Field({}, title="config", description="Configuration settings")
    
class ConfigSchema(TypedDict):
    iconfig: Optional[Dict[str, Any]] = Field({}, title="config", description="Configuration settings")
    

class ToolBaseSchema(BaseModel):
    query: str = Field("", title="query", description="User's question or prompt.")
    code: Optional[str] = Field("", title="code", description="code generated from user input.")
    datatable: Optional[str] = Field("", title="datatable", description="Query result in csv format.")
    response: Optional[str] = Field("", title="response", description="Data analysis insights.")
    info: Optional[dict] = Field({}, title="info", description="Additional information")
    