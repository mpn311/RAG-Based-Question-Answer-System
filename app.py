# app.py
import sys
import types

if "torch.classes" not in sys.modules:
    sys.modules["torch.classes"] = types.ModuleType("torch.classes")

# ----------------- Imports -----------------
import streamlit as st
import chromadb
from chromadb.api.types import EmbeddingFunction
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from PyPDF2 import PdfReader  # for PDF text extraction

NVIDIA_API_KEY = "YOUR_NVIDIA_API_KEY_HERE"  # Replace with your actual NVIDIA API key
NVIDIA_MODEL = "meta/llama3-8b-instruct"
llm = ChatNVIDIA(model=NVIDIA_MODEL, api_key=NVIDIA_API_KEY)

# Custom embedding wrapper for ChromaDB
class CustomNVIDIAEmbeddingFunction(EmbeddingFunction):
    def __init__(self, nvidia_embeddings):
        self.nvidia_embeddings = nvidia_embeddings
    
    def __call__(self, input):
        return self.nvidia_embeddings.embed_documents(input)

# Init embeddings + Chroma
nvidia_embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY)
embedding_function = CustomNVIDIAEmbeddingFunction(nvidia_embeddings)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="docs_collection",
    embedding_function=embedding_function,
    get_or_create=True
)

# ----------------- Helper Functions -----------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def store_in_vector_db(content, source_name):
    # Check if file already exists in collection
    existing = collection.get(where={"source": source_name})
    if existing and existing["ids"]:   # already in DB
        return f"{source_name} already exists in vector DB, skipped."

    # Otherwise, store new chunks
    chunks = split_text(content)
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(
        documents=chunks,
        metadatas=[{"source": source_name} for _ in chunks],
        ids=chunk_ids
    )
    return f"Stored {source_name} in vector DB ✅"

def clear_vector_db():
    global collection
    chroma_client.delete_collection("docs_collection")
    collection = chroma_client.create_collection(
        name="docs_collection",
        embedding_function=embedding_function,
        get_or_create=True
    )

def list_stored_documents():
    docs = collection.get()
    if not docs or not docs["metadatas"]:
        return []
    sources = [meta["source"] for meta in docs["metadatas"] if "source" in meta]
    return sorted(set(sources))  # unique list

def answer_question(question, history):
    # Retrieve relevant chunks
    results = collection.query(query_texts=[question], n_results=3)

    # If no documents found, return fallback
    if not results["documents"] or not results["documents"][0]:
        return "I could not find any relevant information in the uploaded documents."

    retrieved_content = " ".join(results["documents"][0])

    # Format history
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])

    prompt = f"""
    You are a helpful assistant. 
    ONLY use the following context from the uploaded documents to answer. 
    If the answer is not in the context, simply say:
    'The uploaded document does not contain information about this.'

    Context:
    {retrieved_content}

    History:
    {history_text}

    Current Question: {question}
    Answer:
    """
    response = llm.invoke(prompt)
    return response.content

# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="RAG Q&A with History", layout="wide")
    st.title("📚 Document-based Q&A System")

    # Init history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Sidebar showing stored documents
    st.sidebar.header("📂 Stored Documents")
    stored_docs = list_stored_documents()
    if stored_docs:
        for doc in stored_docs:
            st.sidebar.markdown(f"- {doc}")
    else:
        st.sidebar.write("No documents in vector DB")

    # Clear DB button
    if st.sidebar.button("🗑️ Clear All Documents"):
        clear_vector_db()
        st.sidebar.success("Vector DB cleared ✅")

    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF only for now)", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            text = extract_text_from_pdf(uploaded_file)
            msg = store_in_vector_db(text, uploaded_file.name)
            st.success(msg)

    # Q&A Section
    st.subheader("Ask a Question")
    question = st.text_input("Your question:")
    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):
            answer = answer_question(question, st.session_state.qa_history)
            # Save to history
            st.session_state.qa_history.append((question, answer))

    # Display history
    if st.session_state.qa_history:
        st.subheader("Conversation History")
        for q, a in st.session_state.qa_history:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

if __name__ == "__main__":
    main()
