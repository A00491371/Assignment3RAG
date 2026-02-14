# Assignment 2: Document Q&A with RAG

A Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, and **Google Gemini 2.5 Flash** that answers questions based on PDF document content.

## Features

- **PDF Document Loading** — Loads and processes PDFs from the `data/` directory
- **Text Splitting** — Uses `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=200)
- **Vector Store** — ChromaDB for persistent vector storage
- **FREE Embeddings** — HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **RetrievalQA Chain** — LangChain `RetrievalQA` chain connecting retriever with LLM
- **LLM** — Google Gemini 2.5 Flash via `google-genai` SDK
- **Interactive Q&A** — Command-line interface for asking questions
- **Source Citations** — Shows relevant source documents with page numbers

## Setup

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

Get a free API key from: https://aistudio.google.com/app/apikey

### 3. Add Your PDF

Place your PDF in the `data/` directory (e.g., `data/DH-Chapter2.pdf`)

## Usage

### Run Automated Test Queries

```bash
uv run python test_queries.py
```

Results are saved to `output/results.txt`

### Run Interactive Q&A

```bash
uv run python rag_system.py
```

Type your questions and press Enter. Type `quit` to exit.

## How It Works

1. **Document Loading** — PDF is loaded using `PyPDFLoader`
2. **Text Splitting** — Document is split into overlapping chunks
3. **Embedding & Storage** — HuggingFace embeddings are created and stored in ChromaDB
4. **User Query** — User asks a question via command line
5. **Answer Generation** — `RetrievalQA` chain retrieves relevant chunks and Gemini generates an answer

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (FREE) |
| Vector DB | ChromaDB |
| Framework | LangChain (`RetrievalQA` chain) |
| PDF Processing | PyPDF |

## Project Structure

```
Assignment2_RAG/
├── rag_system.py       # Main RAG system with Gemini + RetrievalQA
├── test_queries.py     # Automated test script (3 queries)
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── .env.example        # API key template
├── data/               # PDF documents
│   └── DH-Chapter2.pdf
├── output/             # Test results
│   └── results.txt
└── chroma_db/          # Persistent vector store
```
