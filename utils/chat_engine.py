from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import streamlit as st
from vectorstore.chroma_store import get_vectorstore

@st.cache_resource
def get_llm():
    """Initializes a local LLM for conversational RAG."""
    # Using flan-t5-base as a lightweight instruction-following model
    # It can answer questions based on the provided context
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,
        temperature=0.3,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

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
