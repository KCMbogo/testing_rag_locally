from transformers import pipeline

def classify_query(query: str, llm_pipeline) -> str:
    """
    Classify query as: chitchat, in-domain, or out-of-domain.
    Returns one of the three
    """
    prompt = f"""Classify the following query into exactly one of these categories:
    - chitchat: casual conversation, greetings, small talk
    - in_domain: questions about TANAPA (Tanzania National Parks) and all related information
    - out_of_domain: questions unrelated to TANAPA (Tanzania National Parks) and not casual conversation

    Reply with only the category name, nothing else.

    Query: {query}
    Category:"""

    result = llm_pipeline(prompt, max_new_tokens=10)
    response = result[0]['generated_text'].strip().lower()

    if 'chitchat' in response:
        return 'chitchat'
    elif 'in_domian' in response:
        return 'in_domain'
    elif 'out_of_domain' in response:
        return 'out_of_domain'
    else:
        return 'out_of_domain'