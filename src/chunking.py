from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def semantic_chunk(sentences: list[str], model, threshold: float = 0.5) -> list[str]:
    """
    Split sentences into semantically coherent chunks.
    Cut where similarity between adjascent chunks drops below threshold
    """