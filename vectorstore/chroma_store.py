import os
from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

# Directory to persist ChromaDB
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

@st.cache_resource
def get_embeddings():
    """Gets local lightweight HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore() -> Chroma:
    """Returns the Chroma vectorstore instance."""
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
        
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings()
    )
    return vectorstore

def add_documents_to_store(chunks: List[Any]):
    """Adds document chunks to the vectorstore."""
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=chunks)
    return vectorstore

def similarity_search(query: str, k: int = 4):
    """Performs semantic search on the vectorstore."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results

def clear_vectorstore():
    """Clears the existing vectorstore."""
    try:
        vectorstore = get_vectorstore()
        vectorstore.delete_collection()
    except Exception as e:
        print(f"Error clearing vectorstore: {e}")
