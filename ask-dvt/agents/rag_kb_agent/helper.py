import os, sys, json, textwrap
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)
from state import ToolBaseSchema
from configuration import Configuration
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, Any
import numpy as np

class RAGInput(ToolBaseSchema):
    pass

class RAGOutput(ToolBaseSchema):
    pass

class RAGHelper:
    def __init__(self):
        self.llm = Configuration.query_model
        self.embedding_model = Configuration.embedding_model
    
    def get_llm(self):
        return self.llm
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_agent_prompt(self):
        return Configuration.kb_rag_agent_prompt
    
    def get_tool_base_desc(self):
        tool_base_description = textwrap.dedent("""
            Use this tool to retrieve any information about a big data batch processing platform.

            **Tool Purpose:**
            **Use this tool for any user query related to, but not limited to the following:**

            * **Platform Functionality:** Understanding features, capabilities, and limitations.
            * **Job Management:** Submitting, monitoring, and debugging batch processing jobs.
            * **Data Processing:** Transforming, analyzing, and managing large datasets.
            * **Infrastructure:** Configuring and managing platform resources, components or services.
            * **Troubleshooting:** Diagnosing and resolving errors, performance issues, and unexpected behavior.
            * **Best Practices:** Implementing efficient and optimized workflows.
            * **Code Examples:** Providing sample code snippets for common tasks.
            * **Configuration:** Guidance on configuring various settings, parameters, and services.
            * **Access Control:** Understanding and managing user permissions and roles.
            * **Cost Optimization:** Strategies for reducing operational costs.
            * **API Usage:** Documentation and examples for interacting with the platform's APIs.
            * **Platform Updates:** Information regarding the latest platform changes and releases.
            * **General Inquiries:** Answering any general question related to the platform.
            * **Error Messages:** Explaining error messages and providing solutions.
            * **Performance Tuning:** Optimizing job performance.
            * **Data Security:** Information regarding data security.
            * **Compliance:** Information regarding platform compliance.
            * **Migration:** Helping with platform migration.
            * **Comprehensive dvt Information Retrieval:** Any question or request for information related to the Batch Platform (dvt), regardless of specificity or category. 
            This includes, but is not limited to, the topics listed above, as well as any other aspect of the platform's functionality, configuration, or usage.
        """)
        return tool_base_description
    