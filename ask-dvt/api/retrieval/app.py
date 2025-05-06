import os, sys, re, json, urllib3, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from flask import Flask, jsonify, request
from flask_cors import CORS
from commons.processing import Processor

app = Flask(__name__)
CORS(app)

DEFAULT_CONFIG = {
    "configurable": {
        "thread_id": "46252",
        "num_similar_docs": "8",
        "similarity_threshold": "0.7",
        "query_weight": "0.7"
    }
}

@app.route("/dvt-genai/api/query", methods=["GET", "POST"])
def get_query_response():
    if request.method == "POST":
        data = request.get_json()
        if data and "query" in data:
            query = data["query"]
            start_time = time.time()
            _agent_response_obj = Processor().process_user_query(query, DEFAULT_CONFIG)
            end_time = time.time()
            return {"response": _agent_response_obj.response, "resp_time": f"{end_time - start_time:.2f} seconds"} , 200 #200 OK
        else:
            return "Invalid POST request. Please provide a 'query' in JSON.", 400 #400 Bad Request
    else: #GET
        return "Hello from GET!" , 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001) # Important: host="0.0.0.0"