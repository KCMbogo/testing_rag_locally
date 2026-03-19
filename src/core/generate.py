import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.augment import build_prompt
from core.retrieve import retrieve_chunks
from tools.classifier import classify_query

def load_llm(model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
    """
    Load local LLM and tokenizer.
    device_map='auto' uses GPU if available, falls back to CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, llm


def generate_answer(prompt: str, tokenizer, llm, max_new_tokens: int = 512) -> str:
    """
    Apply chat template, tokenize and generate answer from LLM.
    """
    dialogue = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        dialogue, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def answer_query(query: str, client, collection_name: str,
                 embedding_model, tokenizer, llm, llm_pipeline) -> str:
    """
    Full RAG query pipeline with three-way classification guard.
    """
    # Step 1 - Classify query
    # category = classify_query(query, llm_pipeline)

    # if category == 'chitchat':
    #     # Let LLM respond naturally, no RAG needed
    #     return generate_answer(query, tokenizer, llm)

    # elif category == 'out_of_domain':
    #     return "I'm only able to help with IT support related questions. Please ask something about software configuration or installation."

    # else:
    #     # Full RAG pipeline
    #     retrieved = retrieve_chunks(query, client, collection_name, embedding_model)
    #     print(f"Retrieved: {len(retrieved)} chunks")
    #     for r in retrieved:
    #         print(r.payload['chunk_text'][:100])
    #     prompt = build_prompt(query, retrieved)
    #     return generate_answer(prompt, tokenizer, llm)


    # Full RAG pipeline
    retrieved = retrieve_chunks(query, client, collection_name, embedding_model)
    print(f"Retrieved: {len(retrieved)} chunks")
    for r in retrieved:
        print(r.payload['chunk_text'][:100])
    prompt = build_prompt(query, retrieved)
    return generate_answer(prompt, tokenizer, llm)