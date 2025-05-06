"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.runnables import RunnableConfig, ensure_config
import prompts, vertexai, os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("GOOGLE_CLOUD_REGION")
print(PROJECT_ID, REGION)
print("******************************************")
vertexai.init(project=PROJECT_ID, location=REGION)

@dataclass(kw_only=True)
class Configuration():
    """The configuration for the agent."""

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )
    
    supervisor_agent_prompt: str = field(
        default=prompts.SUPERVISOR_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )
    
    kb_rag_agent_prompt: str = field(
        default=prompts.KB_RAG_AGENT_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )
    
    nlp_to_sql_agent_prompt: str = field(
        default=prompts.NLP_TO_SQL_ANALYST_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )
    
    text_to_psql_converter_prompt: str = field(
        default=prompts.TEXT_TO_PSQL_CONVERTER_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )
    
    data_analysis_expert_prompt: str = field(
        default=prompts.DATA_ANALYSIS_EXPERT_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )

    query_model: Annotated[ChatVertexAI, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=ChatVertexAI(model="gemini-2.0-flash-001", max_tokens=8192, temperature=0.2, additional_kwargs={}),
        metadata={
            "description": "The language model used for processing and refining queries."
        },
    )
    
    embedding_model: Annotated[VertexAIEmbeddings, {"__template_metadata__": {"kind": "embedding"}}] = field(
        default=VertexAIEmbeddings(model="text-embedding-005"),
        metadata={
            "description": "The Text embeddings model used for converting textual data into numerical vectors. These vector representations are designed to capture the semantic meaning and context of the words they represent."
        },
    )
