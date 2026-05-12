import streamlit as st
import time
from components.ui import apply_custom_css, render_card
from utils.pdf_processor import extract_text_from_pdf, process_pdf_for_rag, get_document_stats, clean_text
from utils.summarizer import generate_summary
from vectorstore.chroma_store import add_documents_to_store, clear_vectorstore
from utils.chat_engine import get_conversation_chain

# Configure Streamlit page
st.set_page_config(
    page_title="PDF AI Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for modern UI
apply_custom_css()

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

def handle_upload(uploaded_file):
    """Processes the uploaded PDF file."""
    with st.spinner("Extracting text and analyzing document..."):
        # Extract text
        pdf_bytes = uploaded_file.read()
        text = extract_text_from_pdf(pdf_bytes)
        st.session_state.pdf_text = clean_text(text)
        
        # Get stats
        st.session_state.doc_stats = get_document_stats(text)
        
        # Process for RAG
        st.session_state.vectorstore_ready = False
        clear_vectorstore()
        chunks = process_pdf_for_rag(pdf_bytes, uploaded_file.name)
        add_documents_to_store(chunks)
        st.session_state.vectorstore_ready = True
        
        st.success("Document processed successfully!")
        time.sleep(1)
        st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("📄 AI PDF Hub")
    st.markdown("Upload a document to get started with AI summaries and chat.")
    
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], help="Drag and drop or click to upload")
    
    if uploaded_file is not None and st.session_state.pdf_text == "":
        handle_upload(uploaded_file)
    elif uploaded_file is None:
        # Reset state if file removed
        st.session_state.pdf_text = ""
        st.session_state.summary = ""
        st.session_state.chat_history = []
        st.session_state.vectorstore_ready = False
        
    if st.session_state.pdf_text:
        st.divider()
        st.markdown("### Document Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Words", st.session_state.doc_stats.get("word_count", 0))
        with col2:
            st.metric("Read Time", f"{st.session_state.doc_stats.get('reading_time_mins', 0)} min")
            
        st.divider()
        st.markdown("### Actions")
        if st.button("Clear Document", use_container_width=True):
            st.session_state.pdf_text = ""
            st.session_state.summary = ""
            st.session_state.chat_history = []
            st.session_state.vectorstore_ready = False
            clear_vectorstore()
            st.rerun()

# --- MAIN CONTENT ---
if not st.session_state.pdf_text:
    # Empty State
    st.markdown("<h1 style='text-align: center; color: var(--primary-color); margin-top: 50px;'>Unlock the Power of Your Documents</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: var(--text-color); margin-bottom: 40px;'>Upload a PDF in the sidebar to generate instant summaries, extract key insights, and chat directly with your data.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        render_card("Lightning Fast Summaries", "Get concise overviews of long documents in seconds using local AI models.", "⚡")
    with col2:
        render_card("Interactive Chat", "Ask questions and get answers directly sourced from your document's content.", "💬")
    with col3:
        render_card("Deep Insights", "Extract key bullet points and hidden insights without reading the whole file.", "🧠")

else:
    # Document Loaded State
    tab1, tab2, tab3 = st.tabs(["📑 Summary Generator", "💬 Chat with PDF", "📄 Preview Document"])
    
    with tab1:
        st.markdown("## AI Summarizer")
        
        # Summary Controls
        col1, col2 = st.columns([3, 1])
        with col1:
            summary_type = st.selectbox(
                "Select Summary Type", 
                ["Short Summary", "Detailed Summary", "Bullet Points", "Key Insights"],
                help="Choose how you want the AI to structure the summary."
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate Summary", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing the document..."):
                    # Map UI selection to internal type
                    type_map = {
                        "Short Summary": "short",
                        "Detailed Summary": "detailed",
                        "Bullet Points": "bullet",
                        "Key Insights": "insights"
                    }
                    st.session_state.summary = generate_summary(st.session_state.pdf_text, type_map[summary_type])
        
        # Display Summary
        if st.session_state.summary:
            st.divider()
            st.markdown(f"### Result: {summary_type}")
            with st.container(border=True):
                st.markdown(st.session_state.summary)
            
            # Actions
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                st.download_button(
                    "Download", 
                    st.session_state.summary, 
                    file_name=f"{summary_type.lower().replace(' ', '_')}.txt",
                    use_container_width=True
                )
            # Copy to clipboard is tricky in Streamlit without JS hacks, so we just provide download
            
    with tab2:
        st.markdown("## Chat with your Document")
        st.markdown("Ask anything about the content of the uploaded PDF. The AI will search the document and formulate an answer.")
        
        if st.session_state.vectorstore_ready:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            # Chat input
            if prompt := st.chat_input("Ask a question about the document..."):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        chain = get_conversation_chain()
                        response = chain({"question": prompt})
                        answer = response['answer']
                        
                        # Add citations (optional based on sources)
                        sources = response.get("source_documents", [])
                        if sources:
                            with st.expander("View Source Snippets"):
                                for i, doc in enumerate(sources):
                                    st.markdown(f"**Snippet {i+1}:** {doc.page_content[:200]}...")
                                    
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.info("Document is being processed for chat. Please wait...")

    with tab3:
        st.markdown("## Extracted Text Preview")
        st.markdown("This is the raw text extracted from the PDF, which the AI uses for processing.")
        with st.container(height=500):
            st.text(st.session_state.pdf_text)
