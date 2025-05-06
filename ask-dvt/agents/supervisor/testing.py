from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage

def merge_tool_message_contents(tool_msgs: List[BaseMessage]) -> Dict[str, Any]:
    """
    Iterates over a list of LangChain messages in reverse order, extracts content from ToolMessage instances,
    converts it to dictionaries, and merges them. Values from earlier ToolMessages (later in the list)
    take precedence for common keys.

    Args:
        tool_msgs: A list of LangChain messages.

    Returns:
        A merged dictionary of ToolMessage contents.
    """
    merged_dict: Dict[str, Any] = {}
    
    for message in reversed(tool_msgs):
        if isinstance(message, ToolMessage):
            try:
                content_dict: Dict[str, Any] = eval(message.content) # Safe if content is a valid dictionary string.
                for key, value in content_dict.items():
                    if key not in merged_dict:
                        merged_dict[key] = value
            except (SyntaxError, NameError, TypeError) as e:
                print(f"Error parsing ToolMessage content: {e}")
                #Handle the error as needed, for example, by skipping this message or logging.
                pass #skipping the dictionary that caused the error.
    return merged_dict

# Example usage:
tool_msgs: List[BaseMessage] = [
    AIMessage(content="Hello"),
    HumanMessage(content="What's the weather?"),
    ToolMessage(content='{"temperature": 25, "location": "London"}', tool_name="weather_tool"),
    ToolMessage(content='{"temperature": 20, "humidity": 60, "status": "sunny"}', tool_name="weather_tool"),
    ToolMessage(content='{"location": "Paris", "wind_speed": 10}', tool_name="location_tool"),
]

merged_result = merge_tool_message_contents(tool_msgs)
print(merged_result)