import os
from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Directory to persist ChromaDB
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def get_embeddings():
    """Gets OpenAI embeddings with API key safely."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    return OpenAIEmbeddings(api_key=api_key) if api_key else OpenAIEmbeddings()

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
