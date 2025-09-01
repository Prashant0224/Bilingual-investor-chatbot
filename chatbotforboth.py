import fitz # pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
import os

# --- Configuration ---
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_PATH = "faiss_index.bin"
CHUNKING_SIZE = 500
EMBEDDING_MODEL = None
VECTOR_STORE = None

# --- 1. File Ingestion & Parsing ---
def ingest_and_chunk_pdf(file_path):
    """
    Ingests a PDF file, extracts text, and chunks it.
    Each chunk is tagged with its filename and detected language.
    """
    chunks = []
    if not os.path.exists(file_path):
        print(f"❌ Error ingesting file {file_path}: no such file: '{file_path}'")
        return []
    
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            # Simple chunking by splitting into paragraphs
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50: # Ignore very short chunks
                    # Basic language detection for tagging
                    try:
                        lang = detect(para)
                    except LangDetectException:
                        lang = "unknown"
                    
                    chunks.append({
                        "text": para,
                        "metadata": {
                            "filename": os.path.basename(file_path),
                            "page": page_num + 1,
                            "language": lang
                        }
                    })
        print(f"✅ Ingested {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        print(f"❌ Error ingesting file {file_path}: {e}")
        return []

# --- 2. Embeddings & Vector Store ---
def create_embeddings_and_index(documents):
    """
    Creates text embeddings and stores them in a FAISS index.
    The index is saved to disk for persistence.
    """
    global EMBEDDING_MODEL, VECTOR_STORE
    
    # Load a pre-trained multilingual embedding model
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    print("✅ Multilingual embedding model loaded.")
    
    # Extract all text for embedding
    corpus = [doc['text'] for doc in documents]
    
    # Create embeddings
    print(f"⏳ Creating embeddings for {len(corpus)} documents...")
    embeddings = EMBEDDING_MODEL.encode(corpus, convert_to_numpy=True)
    print("✅ Embeddings created.")

    # Create a FAISS index and add the vectors
    d = embeddings.shape[1]  # Dimension of the embeddings
    VECTOR_STORE = faiss.IndexFlatL2(d)
    VECTOR_STORE.add(embeddings)
    
    # Save the index to disk
    faiss.write_index(VECTOR_STORE, VECTOR_DB_PATH)
    print(f"✅ FAISS index created and saved to {VECTOR_DB_PATH}")

    return VECTOR_STORE, documents

# --- 3. Retrieval + QA Layer ---
def retrieve_and_answer(query, top_k=6):
    """
    Retrieves the most relevant documents from the vector store
    and returns their content.
    """
    global EMBEDDING_MODEL, VECTOR_STORE
    
    if EMBEDDING_MODEL is None or VECTOR_STORE is None:
        print("❌ Vector store not initialized. Please run ingestion first.")
        return "Error: Database not ready."

    # Embed the user's query
    query_embedding = EMBEDDING_MODEL.encode(query, convert_to_numpy=True)
    query_embedding = np.reshape(query_embedding, (1, -1)) # Reshape for FAISS
    
    # Perform a similarity search on the FAISS index
    distances, indices = VECTOR_STORE.search(query_embedding, top_k)
    
    print(f"🔍 Found {len(indices[0])} relevant chunks.")
    
    # Retrieve the actual document content based on the indices
    retrieved_chunks = [documents_corpus[i] for i in indices[0]]
    
    # Construct a simple answer from the retrieved chunks
    answer = "Based on the documents, here is the most relevant information:\n\n"
    for chunk in retrieved_chunks:
        answer += f"--- Source: {chunk['metadata']['filename']} (Page {chunk['metadata']['page']})\n"
        answer += chunk['text'] + "\n\n"

    return answer

# --- Main Chatbot Logic ---
if __name__ == "__main__":
    # Ingest all your target PDF files and combine the results
    documents_corpus = []
    
    # Ingest the financial results PDF
    financial_results_docs = ingest_and_chunk_pdf("FinancialResults_25Q1_J (1) (1).pdf")
    documents_corpus.extend(financial_results_docs)
    
    # Ingest the Sohail Portfolio PDF
    sohail_portfolio_docs = ingest_and_chunk_pdf("Sohail Portfolio.pdf")
    documents_corpus.extend(sohail_portfolio_docs)

    # Exit if no documents were successfully ingested
    if not documents_corpus:
        exit()
    
    # This step will create the FAISS index on disk
    create_embeddings_and_index(documents_corpus)

    # Simple CLI loop for the chatbot
    print("\n--- Mini Bilingual Investor Relations Chatbot (EN/JP) ---")
    print("Type your question in English or Japanese. Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        
        if query.lower() == 'exit':
            print("Chatbot: Goodbye! 👋")
            break

        if not query.strip():
            continue

        try:
            query_lang = detect(query)
            print(f"ℹ️ Detected query language: {query_lang.upper()}")
        except LangDetectException:
            print("ℹ️ Could not detect query language.")

        response = retrieve_and_answer(query)
        print("Chatbot:")
        print(response)