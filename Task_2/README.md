# PDF / Text RAG Pipeline (LangChain + ChromaDB / FAISS)

A Retrieval-Augmented Generation (RAG) pipeline that loads custom PDF and text documents, splits text recursively using `RecursiveCharacterTextSplitter`, generates vector embeddings, and stores them in ChromaDB or FAISS. It retrieves relevant document context to answer user questions accurately using LangChain, with source page citations.

## Project Structure

```text
pdf-rag-pipeline/
├── rag_cli.py
├── requirements.txt
├── .env.example
├── samples/
│   ├── company_policy_handbook.pdf
│   └── architecture_notes.txt
├── src/
│   └── rag_pipeline/
│       ├── __init__.py
│       ├── ingestion.py
│       ├── providers.py
│       └── chain.py
└── tests/
    └── test_rag_pipeline.py
```

## Setup

### 1. Requirements

- Python 3.10 or later
- pip package manager

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure API Keys

Copy `.env.example` to `.env` and configure your preferred API provider:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

Keep your API keys private and do not upload the `.env` file to GitHub.

## Usage

### 1. Index a PDF and Ask a Question Using Gemini and ChromaDB

```powershell
python rag_cli.py `
  --docs samples/company_policy_handbook.pdf `
  --persist-dir .chroma_db `
  --vectorstore chroma `
  --provider gemini `
  --question "How many weeks of paid parental leave do employees receive?"
```

### 2. Query an Existing Vector Database

```powershell
python rag_cli.py --load-only --persist-dir .chroma_db --vectorstore chroma --provider gemini --interactive
```

### 3. Use FAISS and OpenAI with JSON Citation Output

```powershell
python rag_cli.py `
  --docs samples/ `
  --persist-dir .faiss_index `
  --vectorstore faiss `
  --provider openai `
  --question "What is the travel meals daily cap?" `
  --json
```

### 4. Run Offline Using Mock Embeddings and LLM

```powershell
python rag_cli.py --docs samples/company_policy_handbook.pdf --provider mock --vectorstore faiss -q "What is the home office stipend?"
```

## Running Tests

Execute the following command to run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

## Key Features

- Loads custom PDF and text documents.
- Extracts and recursively splits document content.
- Generates vector embeddings.
- Supports ChromaDB and FAISS vector databases.
- Supports Gemini and OpenAI language models.
- Includes offline mock embeddings and LLM support.
- Retrieves relevant context based on user queries.
- Generates answers strictly from retrieved document content.
- Provides source document and page citations.
- Supports command-line and interactive question answering.
- Includes automated tests for document ingestion, retrieval, citations, and CLI functionality.

## Expected Deliverable

A functional RAG pipeline that processes custom PDF and text documents, retrieves relevant information, answers user questions using the provided documents, and displays source citations with page numbers.

## Conclusion

This project demonstrates the implementation of Retrieval-Augmented Generation using Python, LangChain, vector embeddings, and ChromaDB or FAISS. It provides a document-based question-answering system designed to produce accurate, context-aware, and source-supported responses.
