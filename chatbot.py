import os
import fitz  # PyMuPDF
import pdfplumber
import faiss
import pickle
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------
# Config
# ---------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_PATH = "vector_store.faiss"
META_PATH = "vector_store_meta.pkl"
TOP_K = 6   # ✅ retrieve top 6 chunks
TARGET_LANG = "en"  # ✅ only ingest English

# ---------------------------
# Load embedding model
# ---------------------------
print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_text_from_pdf(filepath):
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception:
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text() + "\n"
    return text

# ---------------------------
# Chunk text
# ---------------------------
def chunk_text(text, max_chars=500):
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + " "
        else:
            chunks.append(current.strip())
            current = para + " "
    if current:
        chunks.append(current.strip())
    return chunks

# ---------------------------
# Ingest file (English only)
# ---------------------------
def ingest_file(filepath):
    if filepath.endswith(".pdf"):
        text = extract_text_from_pdf(filepath)
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        return []

    chunks = chunk_text(text)
    english_chunks = []

    for chunk in chunks:
        try:
            lang = detect(chunk)
        except LangDetectException:
            lang = "unknown"

        if lang == TARGET_LANG:   # ✅ keep only English
            english_chunks.append({
                "text": chunk,
                "filename": os.path.basename(filepath),
                "lang": lang
            })

    return english_chunks

# ---------------------------
# Build vector store
# ---------------------------
def build_vector_store(files):
    all_chunks = []
    for f in files:
        all_chunks.extend(ingest_file(f))

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, VECTOR_DB_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Stored {len(all_chunks)} English chunks in FAISS.")

# ---------------------------
# Load vector store
# ---------------------------
def load_vector_store():
    index = faiss.read_index(VECTOR_DB_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ---------------------------
# Retrieve top K
# ---------------------------
def retrieve(query, index, metadata, k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# ---------------------------
# CLI Chatbot
# ---------------------------
def chatbot():
    if not os.path.exists(VECTOR_DB_PATH):
        print("No vector store found. Ingest documents first.")
        return

    index, metadata = load_vector_store()

    print("\n💬 Mini Book Chatbot (English Only)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        results = retrieve(query, index, metadata)
        print("\nBot (retrieved info):")
        for i, res in enumerate(results, 1):
            print(f"[{i}] {res['text']} (from {res['filename']})")
        print("-" * 50)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # ✅ Your book file path
    files = [r"C:\Users\HP\Desktop\Technorizen Task\deeplearningbook.pdf"]

    if not os.path.exists(VECTOR_DB_PATH):
        build_vector_store(files)

    chatbot()
