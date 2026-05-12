import math
import streamlit as st
from transformers import pipeline

@st.cache_resource
def get_summarizer():
    """Loads a lightweight summarization pipeline."""
    # Using t5-small as it is very fast and uses only ~240MB RAM,
    # perfect for free cloud deployment limits!
    return pipeline("summarization", model="t5-small")

def chunk_text(text: str, max_chunk_length: int = 1024) -> list:
    """Splits text into chunks roughly matching the max token length."""
    words = text.split()
    words_per_chunk = int(max_chunk_length / 1.5) 
    
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i:i + words_per_chunk]))
    return chunks

def generate_summary(text: str, summary_type: str = "short") -> str:
    """Generates a summary using local lightweight AI."""
    if not text.strip():
        return "No text provided for summarization."
        
    summarizer = get_summarizer()
    chunks = chunk_text(text, max_chunk_length=400) # Smaller chunks for t5-small
    
    if summary_type == "short":
        max_len = 60
        min_len = 20
    elif summary_type == "detailed":
        max_len = 150
        min_len = 50
    elif summary_type == "bullet":
        max_len = 80
        min_len = 30
    elif summary_type == "insights":
        max_len = 100
        min_len = 40
    else:
        max_len = 130
        min_len = 30

    # Limit chunks to speed up generation on local CPU (max 3 chunks)
    chunks = chunks[:3]
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < min_len:
            summaries.append(chunk)
            continue
            
        try:
            # t5 needs the prefix 'summarize: ' to know what task to do
            t5_chunk = f"summarize: {chunk}"
            
            input_length = len(chunk.split())
            current_max = min(max_len, int(input_length * 0.8))
            current_min = min(min_len, int(input_length * 0.2))
            if current_max <= current_min:
                current_max = current_min + 10
                
            summary = summarizer(t5_chunk, max_length=current_max, min_length=current_min, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append(chunk[:200] + "...") 
            
    combined_summary = " ".join(summaries)
    
    if summary_type == "bullet":
        sentences = combined_summary.split(". ")
        bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
        return "\n".join(bullets)
        
    elif summary_type == "insights":
        sentences = combined_summary.split(". ")
        insights = [f"💡 {s.strip()}" for s in sentences if len(s.split()) > 5]
        return "\n\n".join(insights[:5])
        
    return combined_summary
