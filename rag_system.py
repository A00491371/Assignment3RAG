"""
RAG System using LangChain, ChromaDB, and Google Gemini
Assignment 2: Document Q&A with RAG
"""

import os
from typing import Any, List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from google import genai


class GeminiLLM(LLM):
    """LangChain-compatible wrapper for Google Gemini."""
    client: Any = None
    model_name: str = "gemini-2.5-flash"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"


class RAGSystem:
    def __init__(self, pdf_path: str, persist_directory: str = "chroma_db", use_free_embeddings: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            pdf_path: Path to the PDF document
            persist_directory: Directory to persist ChromaDB
            use_free_embeddings: If True, use free HuggingFace embeddings
        """
        # Load environment variables
        load_dotenv()
        
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.use_free_embeddings = use_free_embeddings
        self.vectorstore = None
        self.qa_chain = None
        
        # Verify Google API key is set
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY not found. Please add your Google API key to .env file.\n"
                "Get one from: https://aistudio.google.com/app/apikey"
            )
        
        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def get_embeddings(self):
        """Get embeddings model (HuggingFace - free)."""
        print("📊 Using FREE HuggingFace embeddings (sentence-transformers)")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def load_documents(self):
        """Load PDF documents from the data directory."""
        print(f"📄 Loading document: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages")
        return documents
    
    def split_documents(self, documents):
        """Split documents into smaller chunks."""
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create and persist ChromaDB vectorstore with embeddings."""
        print("🔢 Creating embeddings and storing in ChromaDB...")
        
        # Get embeddings model
        embeddings = self.get_embeddings()
        
        # Create and persist vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✅ Vectorstore created and persisted to {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self):
        """Load existing vectorstore from disk."""
        print(f"📂 Loading existing vectorstore from {self.persist_directory}...")
        embeddings = self.get_embeddings()
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        print("✅ Vectorstore loaded")
        return self.vectorstore
    
    def create_qa_chain(self):
        """Create the RetrievalQA chain with Google Gemini."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call setup() first.")
        
        print("🔗 Creating RAG chain...")
        
        # Create retriever from vector store
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create Gemini LLM wrapper for LangChain
        llm = GeminiLLM(client=self.gemini_client, model_name="gemini-2.5-flash")
        print("📡 Using Google Gemini 2.5 Flash")
        
        # Use RetrievalQA chain to connect retriever with LLM
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        print("✅ RAG chain created (RetrievalQA)")
    
    def setup(self, force_recreate: bool = False):
        """
        Set up the RAG system.
        
        Args:
            force_recreate: If True, recreate vectorstore even if it exists
        """
        # Check if vectorstore already exists
        if os.path.exists(self.persist_directory) and not force_recreate:
            print(f"ℹ️  Vectorstore already exists at {self.persist_directory}")
            self.load_vectorstore()
        else:
            # Load and process documents
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.create_vectorstore(chunks)
        
        # Create QA chain
        self.create_qa_chain()
        print("🚀 RAG System ready!")
    
    def query(self, question: str, verbose: bool = True):
        """
        Query the RAG system using RetrievalQA chain.
        
        Args:
            question: User's question
            verbose: If True, print source documents
            
        Returns:
            dict: Result containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup() first.")
        
        print(f"\\n❓ Question: {question}")
        
        # Use RetrievalQA chain
        result = self.qa_chain.invoke({"query": question})
        
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        print(f"\\n💡 Answer: {answer}")
        
        if verbose and source_docs:
            print(f"\\n📚 Sources ({len(source_docs)} documents):")
            for i, doc in enumerate(source_docs, 1):
                page = doc.metadata.get("page", "Unknown")
                content_preview = doc.page_content[:200].replace("\\n", " ")
                print(f"\\n  [{i}] Page {page}:")
                print(f"      {content_preview}...")
        
        return {
            "answer": answer,
            "context": source_docs
        }


def main():
    """Main function for interactive Q&A."""
    pdf_path = "data/DH-Chapter2.pdf"
    
    print("\\n" + "="*80)
    print("🤖 RAG System - Powered by Google Gemini 2.5 Flash")
    print("="*80)
    
    rag = RAGSystem(pdf_path, use_free_embeddings=True)
    
    # Setup the system
    rag.setup()
    
    # Interactive Q&A loop
    print("\\n" + "="*80)
    print("🤖 RAG System - Interactive Q&A")
    print("="*80)
    print("Ask questions about the document (or type 'quit' to exit)\\n")
    
    while True:
        question = input("\\n👤 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            rag.query(question)
        except Exception as e:
            print(f"\\n❌ Error: {e}")


if __name__ == "__main__":
    main()
