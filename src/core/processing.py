from unstructured.partition.auto import partition
from unstructured.documents.elements import Title, NarrativeText
from tools.chunking import semantic_chunk
import os

# Load document
def load_document(filepath: str)-> list:
    """
    Load any document format and return structured elements.
    """
    elements = partition(filename=filepath, 
                         languages=["eng", "swa"], 
                         strategy="hi_res", 
                         detect_language_per_element=True,)
    return elements

def group_into_sections(elements: list)-> list[dict]:
    """
    Group document elements into sections based on Title boundaries.
    Each section contains a title and list of sentences under it.
    """
    sections = []
    for item in elements:
        if isinstance(item, Title):
            sections.append({'title': item.text, 'content': []})
        else:
            if any(isinstance(item, dict) for item in sections) and is_clean_text(item.text):
                sections[-1]['content'].append(item.text)
            else:
                if is_clean_text(item.text):
                    sections.append({'title': None, 'content': [item.text]})

    return deduplicate_content(sections)

def is_clean_text(text: str) -> bool:
    if len(text.strip()) < 20:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.6:  # less than 50% actual letters = noise
        return False
    return True

def deduplicate_content(sections: list[dict]) -> list[dict]:
    for section in sections:
        seen = set()
        unique_content = []
        for text in section['content']:
            normalized = " ".join(text.split())
            if normalized not in seen:
                seen.add(normalized)
                unique_content.append(normalized)
        section['content'] = unique_content
    return [s for s in sections if s['content']]  # also removes empty sections

def build_metadata(filepath: str) -> dict:
    """
    Build document level metadata from filepath.
    """
    filename = filepath.split("/")[-1]
    return {
        'filepath': filepath,
        'filename': filename
    }

def index_document(doc_directory: str, embedding_model) -> list[dict]:
    """
    Full indexing pipeline:
    - Load each document
    - Detect structure
    - Semantic chunk per section
    - Embed chunks
    - Return all chunks with metadata
    """
    all_chunks = []
    
    if os.path.isdir(doc_directory):
        for file in os.listdir(doc_directory):
            filepath = os.path.join(doc_directory, file)
            elements = load_document(filepath=filepath)
            sections = group_into_sections(elements=elements)
            doc_metadata = build_metadata(filepath=filepath)

            for section in sections:
                if not section["content"]:
                    continue
                chunks = semantic_chunk(sentences=section['content'])
                for chunk in chunks:
                    all_chunks.append({
                        "section_title": section['title'],
                        "chunk_text": chunk,
                        "metadata": doc_metadata
                    })

        # Embedding chunks at once
        texts = [item['chunk_text'] for item in all_chunks]
        embeddings = embedding_model.encode(texts, show_progress_bar=True)

        for i, item in enumerate(all_chunks):
            item['embedding'] = embeddings[i].tolist()

        return all_chunks
    else:
        print(f"No such directory: {doc_directory}")
        return [{}]


if __name__ == "__main__":
    filepath = "raw_data/arusha.pdf"
    elements = load_document(filepath=filepath)
    sections = group_into_sections(elements=elements)
    print(sections)


