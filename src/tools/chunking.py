from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

def semantic_chunk(sentences: list[str], model=embedding_model, threshold: float = 0.5) -> list[str]:
    """
    Split sentences into semantically coherent chunks.
    Cut where similarity between adjascent chunks drops below threshold
    """
    if len(sentences) <= 1:
        return sentences
    
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

