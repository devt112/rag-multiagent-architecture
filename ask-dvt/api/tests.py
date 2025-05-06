import requests
import json


query = "Can you please share any solution or troubleshooting steps for the following issue: "
query = query + "How to handle repeated events to avoid congestion?. Search Hybrid Responses only."
test_data = {"query": query}

response = requests.post("http://127.0.0.1:5001/dvt-genai/api/query", json=test_data)
print(response.text)