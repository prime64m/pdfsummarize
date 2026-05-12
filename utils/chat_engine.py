import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import streamlit as st
from vectorstore.chroma_store import get_vectorstore

@st.cache_resource
def get_llm():
    """Initializes OpenAI LLM for conversational RAG."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        
    if not api_key:
        st.error("Missing OPENAI_API_KEY! Please set it in your environment or Streamlit Secrets.")
        st.stop()
        
    return ChatOpenAI(
        temperature=0.3,
        model_name="gpt-4o-mini",
        api_key=api_key
    )

def get_conversation_chain():
    """Creates the conversational retrieval chain."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain
