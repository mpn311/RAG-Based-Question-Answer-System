# 📚 RAG Q&A with NVIDIA + ChromaDB + Streamlit  

A **Retrieval Augmented Generation (RAG)** system that enables users to upload PDF documents and ask natural language questions. The system retrieves relevant document chunks using **NVIDIA embeddings + ChromaDB**, and generates accurate answers with **NVIDIA LLMs** — all inside a simple **Streamlit web app**.  

---

## 🚀 Features  
- 🔹 **PDF Upload & Processing** – Extracts text from PDF documents.  
- 🔹 **Text Chunking** – Splits text into manageable chunks (1000 chars, 200 overlap).  
- 🔹 **Semantic Search** – Stores chunks in **ChromaDB** with NVIDIA embeddings.  
- 🔹 **RAG Q&A** – Uses retrieved chunks as context for **NVIDIA LLM** (`meta/llama3-8b-instruct`).  
- 🔹 **Conversation History** – Maintains Q&A history for continuity.  
- 🔹 **Streamlit UI** – Clean interface with document listing, clearing, and Q&A.  
- 🔹 **Duplicate Handling** – Skips already uploaded documents.  
- 🔹 **Fallback Safety** – If no answer found, responds with a clear fallback.  

---


