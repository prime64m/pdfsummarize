import os
import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import pypdf
import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
        return ""

def process_pdf_for_rag(pdf_bytes: bytes, filename: str) -> List[Any]:
    """Saves PDF to a temporary file, loads it via LangChain, and chunks it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Add filename metadata
        for doc in documents:
            doc.metadata["source"] = filename
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    finally:
        os.remove(tmp_path)

def get_document_stats(text: str) -> Dict[str, Any]:
    """Calculates word count and reading time."""
    words = len(text.split())
    # Average reading speed: 200 words per minute
    reading_time_minutes = max(1, round(words / 200))
    
    return {
        "word_count": words,
        "reading_time_mins": reading_time_minutes
    }

def clean_text(text: str) -> str:
    """Basic text cleaning for summarization."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might break the summarizer
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text.strip()
