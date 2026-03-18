import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU entirely

from core.generate import answer_query
from core.db import qdrant_client
from index import embedding_model, tokenizer, llm, llm_pipeline

# --- QUERY (runs every time user asks) ---

query = "What are the core values of TANAPA"
response = answer_query(
    query,
    qdrant_client,
    "tanapa_data",
    embedding_model,
    tokenizer,
    llm,
    llm_pipeline
)
print(f"Query: {query}\n\nResponse: {response}")