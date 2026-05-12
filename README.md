# AI-Powered PDF Summarizer & Chat Hub

A modern, SaaS-style web application built with Streamlit, Python, and Local AI Models (HuggingFace, LangChain, ChromaDB). This application allows users to upload PDF documents, generate different types of AI summaries, extract insights, and interactively chat with the document contents using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Modern SaaS UI**: Dark/light mode compatible, animated loaders, clean typography, and responsive cards.
- **Instant Summaries**: Generate short summaries, detailed summaries, bullet points, and extract key insights using local `distilbart-cnn-12-6`.
- **Chat with PDF**: Ask questions about your document using a conversational retrieval chain backed by `flan-t5-base` and ChromaDB vector search.
- **Privacy-First**: Uses local HuggingFace pipelines. No data is sent to external APIs like OpenAI unless explicitly configured.
- **Document Analytics**: Instantly see reading time, word count, and a full text preview of the uploaded document.

## Project Architecture

```
project/
├── app.py                     # Main Streamlit application
├── components/                # UI components
│   └── ui.py                  # Custom CSS injection and styled cards
├── utils/                     # Utility modules
│   ├── pdf_processor.py       # PDF parsing, chunking, and text extraction
│   ├── summarizer.py          # HuggingFace local summarization pipeline
│   └── chat_engine.py         # Conversational RAG chain using Langchain
├── vectorstore/               # Vector database modules
│   └── chroma_store.py        # Local ChromaDB operations and embeddings setup
├── requirements.txt           # Dependencies
└── .env                       # Environment variables template
```

## Setup Instructions

1. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Since this app uses local HuggingFace models, the initial run will download the model weights (approx 1.5GB total).*

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open your browser to `http://localhost:8501`.

## Technologies Used
- Frontend: Streamlit
- PDF Extraction: PyPDF2, PyPDFLoader
- AI Pipelines: HuggingFace Transformers (`sshleifer/distilbart-cnn-12-6`, `google/flan-t5-base`)
- Embeddings: `all-MiniLM-L6-v2` via Sentence-Transformers
- RAG Pipeline: LangChain
- Vector Database: ChromaDB
