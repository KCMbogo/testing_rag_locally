# --- INDEXING (run once) ---
from transformers import pipeline

from core.generate import answer_query, load_llm
from processing import index_document
from storage import setup_qdrant, store_chunks
from tools.chunking import embedding_model


all_chunks = index_document("raw_data/", embedding_model)
qdrant_client = setup_qdrant("it_support_docs")
store_chunks(qdrant_client, "it_support_docs", all_chunks)

# --- LOAD LLM (run once) ---
tokenizer, llm = load_llm("paraphrase-multilingual-MiniLM-L12-v2")
llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)

# --- QUERY (runs every time user asks) ---
query = "What is the long form of TANAPA"
response = answer_query(
    query,
    qdrant_client,
    "it_support_docs",
    embedding_model,
    tokenizer,
    llm,
    llm_pipeline
)
print(response)