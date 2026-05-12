import math
import streamlit as st
from transformers import pipeline

@st.cache_resource
def get_summarizer():
    """Loads the summarization pipeline and caches it to avoid reloading."""
    # Using a fast and small model suitable for local CPU/GPU execution
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def chunk_text(text: str, max_chunk_length: int = 1024) -> list:
    """Splits text into chunks roughly matching the max token length."""
    words = text.split()
    # Assuming average of 1.5 tokens per word for a simple heuristic
    words_per_chunk = int(max_chunk_length / 1.5) 
    
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i:i + words_per_chunk]))
    return chunks

def generate_summary(text: str, summary_type: str = "short") -> str:
    """
    Generates a summary of the provided text.
    Handles long texts by chunking, summarizing chunks, and combining them.
    """
    if not text.strip():
        return "No text provided for summarization."
        
    summarizer = get_summarizer()
    chunks = chunk_text(text, max_chunk_length=800) # conservative chunk size
    
    # Configure length based on type
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

    # Limit chunks to speed up generation on local CPU (max 3 chunks / ~1500 words)
    chunks = chunks[:3]
    summaries = []
    for chunk in chunks:
        # If the chunk is very short, just append it
        if len(chunk.split()) < min_len:
            summaries.append(chunk)
            continue
            
        try:
            # max_length cannot exceed input length
            input_length = len(chunk.split())
            current_max = min(max_len, int(input_length * 0.8))
            current_min = min(min_len, int(input_length * 0.2))
            
            # Ensure max > min
            if current_max <= current_min:
                current_max = current_min + 10
                
            summary = summarizer(chunk, max_length=current_max, min_length=current_min, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append(chunk[:200] + "...") # Fallback
            
    combined_summary = " ".join(summaries)
    
    # Post-processing based on type
    if summary_type == "bullet":
        # Split into sentences and format as bullets
        sentences = combined_summary.split(". ")
        bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
        return "\n".join(bullets)
        
    elif summary_type == "insights":
        # Extract key insights (simulated by taking longer/important-looking sentences)
        sentences = combined_summary.split(". ")
        insights = [f"💡 {s.strip()}" for s in sentences if len(s.split()) > 5]
        return "\n\n".join(insights[:5]) # Top 5 insights
        
    return combined_summary
