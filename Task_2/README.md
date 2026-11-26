# PDF / Text RAG Pipeline
### Retrieval-Augmented Generation Using LangChain, ChromaDB, and FAISS

## Project Overview

The **PDF / Text RAG Pipeline** is a document-based question-answering system built with Python and LangChain. It implements Retrieval-Augmented Generation (RAG) to retrieve relevant information from custom PDF and text documents and generate context-aware answers based on the provided content.

The pipeline extracts document content, divides it into manageable text chunks using `RecursiveCharacterTextSplitter`, generates vector embeddings, and stores them in a vector database such as ChromaDB or FAISS. When a user submits a question, the system retrieves the most relevant document segments and uses them as context for generating an accurate response.

The system also supports source attribution and page-level citations to help users verify the information provided.

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

## Technologies Used

- **Python:** Core programming language.
- **LangChain:** Document processing and RAG workflow orchestration.
- **ChromaDB:** Vector database for storing and retrieving embeddings.
- **FAISS:** Alternative vector similarity search and storage.
- **Gemini API:** Large language model integration.
- **OpenAI API:** Alternative language model and embedding provider.
- **Pydantic:** Structured data validation, where applicable.
- **PyPDF:** PDF document loading and text extraction.

## System Workflow

1. **Document Loading:** Load PDF and text files using appropriate document loaders.
2. **Text Extraction:** Extract readable content and document metadata.
3. **Text Chunking:** Split extracted content into smaller sections using `RecursiveCharacterTextSplitter`.
4. **Embedding Generation:** Convert text chunks into numerical vector representations.
5. **Vector Storage:** Store generated embeddings in ChromaDB or FAISS.
6. **Similarity Search:** Retrieve document chunks relevant to the user's question.
7. **Answer Generation:** Provide retrieved context to the language model to generate a grounded response.
8. **Source Attribution:** Include source document information and page references where available.

## Setup and Installation

### 1. Prerequisites

Make sure the following are installed:

- Python 3.10 or later
- pip package manager

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file by copying `.env.example` and add the API key for your preferred provider:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

Keep your API keys secure. Never upload your `.env` file or expose API credentials in a public repository.

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

Load an existing ChromaDB index and start an interactive question-answering session:

```bash
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

### 4. Run in Offline Mock Mode

Run the pipeline using mock embeddings and a mock language model without requiring an external API key:

```bash
python rag_cli.py --docs samples/company_policy_handbook.pdf --provider mock --vectorstore faiss -q "What is the home office stipend?"
```

## Running Tests

Execute the following command to run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

The tests are intended to verify document ingestion, retrieval behavior, source citations, and command-line functionality.

## Key Features

- Supports PDF and plain-text document ingestion.
- Extracts and processes document content and metadata.
- Implements recursive text splitting for efficient document chunking.
- Generates vector embeddings for semantic search.
- Supports ChromaDB and FAISS vector stores.
- Integrates with Gemini and OpenAI language models.
- Provides offline mock support for testing.
- Retrieves relevant context using vector similarity search.
- Generates answers grounded in the retrieved document content.
- Provides source document references and page citations where available.
- Supports command-line and interactive question-answering modes.
- Includes automated tests for core pipeline functionality.

## Expected Deliverable

A functional Retrieval-Augmented Generation pipeline capable of:

- Loading and processing custom PDF and text documents.
- Generating and storing vector embeddings.
- Retrieving relevant document sections based on user queries.
- Generating context-aware answers using retrieved information.
- Providing source references and page citations to support generated answers.

## Conclusion

This project demonstrates the practical implementation of a Retrieval-Augmented Generation system using Python, LangChain, vector embeddings, and modern vector databases. By combining document processing, semantic retrieval, and large language models, the pipeline provides a structured approach to document-based question answering.

It aims to improve answer relevance and traceability by grounding responses in the supplied documents and providing source references for verification.
