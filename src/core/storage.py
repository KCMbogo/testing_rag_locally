from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def setup_qdrant(collection_name: str, vector_size: int = 384) -> QdrantClient:
    client = QdrantClient(path="qdrant_db")

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    return client


def store_chunks(client, collection_name: str, all_chunks: list[dict]):
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={
                'chunk_text': chunk['chunk_text'],
                'section_titile': chunk['section_title'],
                'filename': chunk['metadata']['filename'],
                'filepath': chunk['metadata']['filepath']
            }
        ))

    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} chunks in Qdrant")








