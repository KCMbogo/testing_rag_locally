def build_prompt(query: str, retrieved_chunks: list) -> str:
    context = "\n\n".join([
        f"[Source: {r.payload['filename']} | Section: {r.payload['section_titile']}]\n{r.payload['chunk_text']}"
        for r in retrieved_chunks
    ])

    prompt = f"""You are a helpful TANAPA (Tanzania National Parks) assistant. Answer the question using only the context provided below.
    If the answer is not in the context, say you don't know.
    Be clear and explanatory in your answer.

    Context:
    {context}

    Question: {query}
    Answer:"""

    return prompt