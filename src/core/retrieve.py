def retrieve_chunks(query: str, client, collection_name: str,
                    embedding_model, n_results: int = 5) -> list[dict]:
    """
    Embed query and retrieve top n most similar chunks ffom Qdrant
    """
    query_embedding = embedding_model.encode(query).tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=n_results,
        with_payload=True,
    )

    return results