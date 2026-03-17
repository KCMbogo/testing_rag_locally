# --- INDEXING (run once) ---
from transformers import pipeline

from core.generate import load_llm
from processing import index_document
from storage import setup_qdrant, store_chunks
from tools.chunking import embedding_model


all_chunks = index_document("raw_data/", embedding_model)
qdrant_client = setup_qdrant("tanapa_data")
store_chunks(qdrant_client, "tanapa_data", all_chunks)

# --- LOAD LLM (run once) ---
tokenizer, llm = load_llm("paraphrase-multilingual-MiniLM-L12-v2")
llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)

print(f"The loaded model: {llm}")