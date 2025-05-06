import os, sys, json, textwrap, urllib3, time, yaml

def read_actions_from_yaml(yaml_data):
    try:
        data = yaml.safe_load(yaml_data)
        actions = data.get('runbook', {}).get('actions', [])
        action_names = [action['name'] for action in actions if 'name' in action]
        return actions, action_names
    except yaml.YAMLError as e:
        print(f"Error reading YAML: {e}")
        return []


def get_runbook():
    with open(r"C:\dev\99999_DVT_ASKDVT_AI_MULTIAGENT\ask-dvt\agents\hil_runbook\sample_runbook.yml", 'r') as file:
        return file.read()


selected_action = "Create User"
runbook = get_runbook()
_actions_, _actions_list_ = read_actions_from_yaml(runbook)
_action_parameters_ = next((action.get('parameters', []) for action in _actions_ if action.get('name') == selected_action), None)
text_parameters = [{
    "name": p.get("name"), 
    "display_name": p.get("display_name"), 
    "description": p.get("description"), 
    "secret": p.get("secret", False)
} for p in _action_parameters_ if p.get('type') == 'text']

choice_parameters = [{"name": p.get("name"), 
                     "description": p.get("description"),
                     "options": p.get("options", [])  # Ensure "options" is a list
                     } 
                    for p in _action_parameters_ if p.get("type") == "choice"]
print(choice_parameters)