from transformers import pipeline

from core.generate import load_llm
from core.processing import index_document
from core.storage import setup_qdrant, store_chunks
from tools.chunking import embedding_model


# --- INDEXING (run once) ---
import os
if not os.path.exists('qdrant_db/'):
    all_chunks = index_document("raw_data/", embedding_model)
    qdrant_client = setup_qdrant("tanapa_data")
    store_chunks(qdrant_client, "tanapa_data", all_chunks)

# --- LOAD LLM (run once) ---
tokenizer, llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3")
llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)

print(f"The loaded model: {llm}")