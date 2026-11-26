# PDF / Text RAG Pipeline (LangChain + ChromaDB / FAISS)

Grounded Retrieval-Augmented Generation (RAG) pipeline that loads custom PDF and text documents, splits text recursively with `RecursiveCharacterTextSplitter`, indexes vector embeddings in **ChromaDB** or **FAISS**, and answers user queries strictly from retrieved context using LangChain `RetrievalQA` with source page citations.

## Project Structure

```text
pdf-rag-pipeline/
├── rag_cli.py                          # Executable CLI & interactive Q&A loop
├── requirements.txt                    # LangChain, ChromaDB, FAISS, PyPDF dependencies
├── .env.example                        # API key and model configuration template
├── samples/
│   ├── company_policy_handbook.pdf     # Sample 3-page PDF document
│   └── architecture_notes.txt          # Sample plain-text document
├── src/rag_pipeline/
│   ├── __init__.py
│   ├── ingestion.py                    # PyPDFLoader/TextLoader, RecursiveCharacterTextSplitter, Chroma/FAISS
│   ├── providers.py                    # Gemini, OpenAI, and deterministic offline mock embeddings/LLMs
│   └── chain.py                        # Strictly grounded RetrievalQA chain + page citations
└── tests/
    └── test_rag_pipeline.py            # End-to-end ingestion, vector store, citation, and CLI tests
```

## Setup

1. Ensure Python 3.10+ is installed, then create a virtual environment and install dependencies:

```powershell
cd "C:\Users\Saif Ullah\.gemini\antigravity\scratch\pdf-rag-pipeline"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` or `OPENAI_API_KEY`.

## Usage

### Index a Custom PDF in ChromaDB & Ask a Question (Gemini)

```powershell
python rag_cli.py `
  --docs samples/company_policy_handbook.pdf `
  --persist-dir .chroma_db `
  --vectorstore chroma `
  --provider gemini `
  --question "How many weeks of paid parental leave do employees receive?"
```

### Query an Existing Persisted Index Without Re-Indexing

```powershell
python rag_cli.py --load-only --persist-dir .chroma_db --vectorstore chroma --provider gemini --interactive
```

### Use FAISS & OpenAI with JSON Citation Output

```powershell
python rag_cli.py `
  --docs samples/ `
  --persist-dir .faiss_index `
  --vectorstore faiss `
  --provider openai `
  --question "What is the travel meals daily cap?" `
  --json
```

### Run Offline with Mock Embeddings & LLM

```powershell
python rag_cli.py --docs samples/company_policy_handbook.pdf --provider mock --vectorstore faiss -q "What is the home office stipend?"
```

## Running Tests

```powershell
python -m unittest discover -s tests -v
```
