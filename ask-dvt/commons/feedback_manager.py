import os, sys, re, json, urllib3
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from db_helper import pgdbWrapper
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, Any

import numpy as np  # Ensure numpy is imported

class FeedbackManager:
    def __init__(self):
        self.db = pgdbWrapper()

    @classmethod
    def embed_text(cls, text):
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        embeddings = model.get_embeddings([text])
        return np.array(embeddings[0].values)

    def save_feedback(self, feedback_data: dict, table_name="approved_responses"):
        columns, values, params = [], [], []
        for key, value in feedback_data.items():
            if "embeddings" in key:
                for embed_key, embed_value in value.items():
                    columns.append(embed_value)
                    values.append("%s")
                    params.append(self.embed_text(feedback_data[embed_key]))
            else:
                columns.append(key)
                values.append("%s")
                params.append(value)

        columns_str = ", ".join(columns)
        values_str = ", ".join(values)
        sql_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_str})"
        result = self.db.execute_insert_or_update(sql_query, params)
        if result: print("Feedback saved successfully.")
        return result