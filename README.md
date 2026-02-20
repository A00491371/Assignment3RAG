# Assignment 3: Secure & Production-Ready RAG System

This project is an evolution of the Assignment 2 RAG system, enhanced with production-grade security layers, guardrails, and automated evaluation. It uses **LangChain**, **ChromaDB**, and **Google Gemini 2.5 Flash** to provide a secure Q&A experience based on the Nova Scotia Driver's Handbook.

## New Features (Assignment 3)

### 1. Robust Guardrails
The system implements multiple layers of input and output validation to ensure safety and reliability:
- **Input Validation**: Rejects queries over 500 characters and filters out-of-topic questions.
- **PII Protection**: Automatically detects and redacts phone numbers, emails, and license plates.
- **Output Controls**: Enforces word limits on responses and refuses to answer when retrieval confidence is too low ($top\_score < 0.1$).
- **Structured Error Handling**: Returns specific error codes like `QUERY_TOO_LONG`, `OFF_TOPIC`, `PII_DETECTED`, etc.
- **Execution Limits**: Implemented a 30-second timeout for LLM generation to prevent runaway processes.

### 2. Prompt Injection Defenses
I implemented the following 3 primary defenses against adversarial attacks:
1.  **Input Sanitization**: Scans for and neutralizes patterns like "ignore previous instructions", "you are now", or "### system:" before they reach the model.
2.  **Instruction-Data Separation**: Uses clear XML-style delimiters (`<retrieved_context>`) to separate instructions from external document data, helping the LLM distinguish between ground truth and potential data-driven injections.
3.  **System Prompt Hardening**: A strictly defined system persona that explicitly instructs the model to treat retrieved data as untrusted and never reveal its internal instructions. I also implemented **Output Validation** as a 4th layer to catch any leakage if instructions are mentioned in the response.

### 3. Evaluation Metric: Faithfulness
- **Metric Chosen**: **Faithfulness (Groundedness)**.
- **Why**: In a regulatory and safety-critical domain like driving rules, the most important factor is that the AI does not hallucinate. Faithfulness measures whether every claim in the answer is directly supported by the retrieved document chunks.
- **Implementation**: Uses an LLM-as-a-judge approach to verify compliance for every query.

## Test Results & Dashboard

The following summary is generated after running the complete test suite (9 scenarios including normal queries, injections, and off-topic cases):

```text
==================== DASHBOARD SUMMARY ====================
Total Queries Processed: 9
Guardrails Triggered:
  - INJECTION_ATTEMPT: 3
  - OFF_TOPIC: 1
  - PII_DETECTED: 1
Injection Attempts Blocked: 3
Average Faithfulness Score (Yes/No ratio): 1.00
============================================================
```

## Interesting Findings from Test Results
- **Resilience to Jailbreaks**: All 3 test attacks (including "ignore instructions" and "print system prompt") were successfully intercepted. The sanitization layer caught the patterns, and the policy guardrail blocked the generation.
- **High Groundedness**: The system maintained a **1.00 average faithfulness score** during testing, indicating that the use of strict instruction-data separation effectively prevents the model from "making things up" outside the provided context.
- **Low-Score Precision**: Setting a retrieval threshold (0.1) successfully prevented the model from trying to answer questions about chocolate cake or other irrelevant topics that slipped past the initial off-topic classifier.

## Setup & Usage

### 1. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run Secure Tests
```bash
uv run python secure_rag.py
```
This runs the full suite of normal, injection, and off-topic test cases and generates a dashboard summary in `output/results.txt`.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Guardrails | Custom Python logic + Pattern Matching |
| Evaluation | LLM-as-a-Judge (Faithfulness) |

## Project Structure

```
.
├── rag_system.py       # Core RAG logic (Assignment 2 base)
├── secure_rag.py       # Security & Evaluation layer (Assignment 3)
├── data/               # PDF source documents
├── output/             # Results & Logs
│   ├── results.txt     # Test report & Dashboard summary
│   └── guardrails.log  # Trigger alerts
└── chroma_db/          # Persistent vector store
```


