import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU entirely

from core.generate import answer_query
from core.db import qdrant_client
from index import embedding_model, tokenizer, llm, llm_pipeline

# --- QUERY (runs every time user asks) ---
import time

query = "Hello, who are you? And tell me abit about Bitcoin."

start_time = time.time()
response = answer_query(
    query,
    qdrant_client,
    "tanapa_data",
    embedding_model,
    tokenizer,
    llm,
    llm_pipeline
)
end_time = time.time()
time_taken = end_time - start_time
print(f"Query: {query}\n\nResponse: {response}\n\n Time taken: {time_taken: .2f}")