# Bilingual-RAG-chatbot
 📊 Bilingual Investor Relations Chatbot (EN/JP)

An AI-powered chatbot that allows users to query investor-related PDF documents (like financial reports or portfolios) in English and Japanese.
It uses PyMuPDF (fitz) for PDF parsing, Sentence-Transformers for embeddings, FAISS for vector search, and LangDetect for automatic language detection.

🚀 Features

📂 PDF Ingestion & Chunking: Automatically extracts and chunks text from uploaded PDFs.

🌍 Language Detection: Detects the language (English, Japanese, or unknown) of each chunk and user query.

🤖 Multilingual Embeddings: Uses paraphrase-multilingual-MiniLM-L12-v2 to support multiple languages.

⚡ Vector Search with FAISS: Efficient similarity-based retrieval of relevant document chunks.

💬 Interactive Chatbot: Query in English or Japanese and get contextually relevant answers.

🛠️ Tech Stack

Python 3.10+

PyMuPDF (fitz)
 → PDF parsing

Sentence-Transformers
 → Text embeddings

FAISS
 → Vector database

LangDetect
 → Language detection

NumPy, OS

📂 Project Structure

📦 bilingual-investor-chatbot

 ┣ 📜 main.py              # Main chatbot script
 
 ┣ 📜 requirements.txt     # Dependencies
 
 ┣ 📜 FinancialResults.pdf # Sample document (if allowed)
 
 ┣ 📜 SohailPortfolio.pdf  # Sample document (if allowed)
 
 ┣ 📜 README.md            # Project description
 
 ┗ 📦 vector_store.faiss   # Saved FAISS index

 ▶️ Usage

Place your PDF documents in the project directory.

Run the chatbot:

python main.py


Ask questions in English or Japanese:

You: What was the Q1 revenue in 2025?

You: 2025年のQ1の収益は？



