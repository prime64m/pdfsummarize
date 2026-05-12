import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

@st.cache_resource
def get_summarizer():
    # Attempt to load from env or streamlit secrets
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        
    if not api_key:
        st.error("Missing OPENAI_API_KEY! Please set it in your environment or Streamlit Secrets.")
        st.stop()
        
    return ChatOpenAI(temperature=0.3, model_name="gpt-4o-mini", api_key=api_key)

def generate_summary(text: str, summary_type: str = "short") -> str:
    """Generates a summary using OpenAI."""
    if not text.strip():
        return "No text provided for summarization."
        
    llm = get_summarizer()
    
    # Simple prompt mapping
    prompts = {
        "short": "Provide a brief, concise 2-sentence summary of the following text:\n\n{text}",
        "detailed": "Provide a detailed, comprehensive summary of the following text:\n\n{text}",
        "bullet": "Summarize the following text using bullet points:\n\n{text}",
        "insights": "Extract the top 5 key insights from the following text:\n\n{text}"
    }
    
    prompt = PromptTemplate.from_template(prompts.get(summary_type, prompts["short"]))
    
    try:
        # Send the whole text up to a safe limit for gpt-4o-mini
        safe_text = text[:100000]
        chain = prompt | llm
        result = chain.invoke({"text": safe_text})
        return result.content
    except Exception as e:
        print(f"Error summarizing with OpenAI: {e}")
        return f"Error generating summary: {e}"
