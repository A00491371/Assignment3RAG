"""
Test script to run the three required queries and save results.
Uses FREE HuggingFace embeddings to avoid OpenAI embedding costs.
"""

import os
from rag_system import RAGSystem


def main():
    # Test queries from assignment
    queries = [
        "What is Crosswalk guards?",
        "What to do if moving through an intersection with a green signal?",
        "What to do when approached by an emergency vehicle?"
    ]
    
    # Initialize RAG system with FREE embeddings
    print("🚀 Initializing RAG System with FREE HuggingFace embeddings...")
    pdf_path = "data/DH-Chapter2.pdf"
    rag = RAGSystem(pdf_path, use_free_embeddings=True)
    rag.setup()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Run queries and save results
    results_file = "output/results.txt"
    print(f"\\n📝 Running test queries and saving to {results_file}...")
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\\n")
        f.write("RAG SYSTEM TEST RESULTS\\n")
        f.write("Assignment 2: Document Q&A with RAG\\n")
        f.write("Using FREE HuggingFace Embeddings (all-MiniLM-L6-v2)\\n")
        f.write("=" * 80 + "\\n\\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\\n{'='*80}")
            print(f"Query {i}/{len(queries)}")
            
            # Query the system
            result = rag.query(query, verbose=True)
            
            # Write to file
            f.write(f"Query {i}: {query}\\n")
            f.write("-" * 80 + "\\n")
            f.write(f"Answer: {result['answer']}\\n\\n")
            
            # Write source documents
            f.write("Source Documents:\\n")
            context_docs = result.get('context', [])
            for j, doc in enumerate(context_docs, 1):
                page = doc.metadata.get('page', 'Unknown')
                content = doc.page_content[:300].replace('\\n', ' ')
                f.write(f"\\n  [{j}] Page {page}:\\n")
                f.write(f"  {content}...\\n")
            
            f.write("\\n" + "=" * 80 + "\\n\\n")
    
    print(f"\\n✅ Results saved to {results_file}")
    print(f"\\n🎉 Test completed successfully!")


if __name__ == "__main__":
    main()
