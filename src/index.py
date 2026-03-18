from transformers import pipeline

from core.generate import load_llm
from core.processing import index_document
from core.storage import store_chunks
from core.db import qdrant_client
from tools.chunking import embedding_model


# --- INDEXING (run once) ---
import os
if not os.path.exists('qdrant_db/'):
    all_chunks = index_document("raw_data/", embedding_model)
    store_chunks(qdrant_client, "tanapa_data", all_chunks)

# --- LOAD LLM (run once) ---
tokenizer, llm = load_llm("microsoft/Phi-3-mini-4k-instruct")
llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)

print(f"The loaded model: {llm}")