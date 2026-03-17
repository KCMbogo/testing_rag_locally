from core.generate import answer_query
from .index import qdrant_client, embedding_model, tokenizer, llm, llm_pipeline


# --- QUERY (runs every time user asks) ---

query = "What is the long form of TANAPA"
response = answer_query(
    query,
    qdrant_client,
    "tanapa_data",
    embedding_model,
    tokenizer,
    llm,
    llm_pipeline
)
print(response)