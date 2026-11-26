# Structured Document Analyzer (Gemini / OpenAI + Pydantic)

A Python-based document analysis pipeline that converts noisy, unstructured invoices and resumes into structured, validated JSON data.

The system uses parameterized few-shot prompt templates, Google Gemini or OpenAI APIs, and Pydantic v2 for data validation. It also includes an automatic validation and correction mechanism to handle invalid model responses.

## Project Structure

```text
structured-doc-analyzer/
├── analyze.py
├── requirements.txt
├── .env.example
├── samples/
│   ├── messy_invoice.txt
│   └── messy_resume.txt
├── src/
│   └── doc_analyzer/
│       ├── __init__.py
│       ├── schemas.py
│       ├── prompts.py
│       └── analyzer.py
└── tests/
    └── test_pipeline.py
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Main programming language |
| Google Gemini | AI-powered document analysis |
| OpenAI API | Alternative language model provider |
| Pydantic v2 | Structured data validation |
| Prompt Templates | Parameterized prompt generation |
| Few-Shot Learning | Improve extraction accuracy |
| Python Unittest | Automated testing |

## Setup and Installation

### 1. Requirements

- Python 3.10 or later
- pip package manager
- Gemini API key or OpenAI API key

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Copy `.env.example` to `.env` and configure your preferred API provider.

For Google Gemini:

```env
GEMINI_API_KEY=your_gemini_api_key
```

For OpenAI:

```env
OPENAI_API_KEY=your_openai_api_key
```

Keep your API keys private and never upload the `.env` file to GitHub.

## Usage

### 1. Analyze an Invoice Using Gemini

```bash
python analyze.py --type invoice --input samples/messy_invoice.txt --provider gemini --currency USD
```

### 2. Analyze a Resume Using OpenAI

```bash
python analyze.py --type resume --input samples/messy_resume.txt --provider openai --target-role "Staff ML Platform Engineer" --output resume_out.json
```

### 3. Inspect the Generated Prompt

```bash
python analyze.py --type resume --input samples/messy_resume.txt --target-role "Backend Engineer" --show-prompt
```

### 4. Run Offline Using Mock Provider

```bash
python analyze.py --type invoice --input samples/messy_invoice.txt --provider mock
```

## Running Tests

Run the following command to execute the automated test suite:

```bash
python -m unittest discover -s tests -v
```

## Key Features

- Extracts structured information from unstructured invoices and resumes.
- Supports Google Gemini and OpenAI API providers.
- Uses parameterized prompt templates.
- Includes few-shot examples for improved extraction.
- Validates generated data using Pydantic v2 schemas.
- Performs cross-field validation.
- Includes automatic validation and correction attempts.
- Produces structured JSON output.
- Supports offline testing through a mock provider.
- Includes automated tests for schemas, prompts, validation, and CLI functionality.

## Expected Deliverable

A functional Python application that analyzes unstructured invoices and resumes and converts their information into strictly validated JSON output using Gemini or OpenAI.

## Conclusion

This project demonstrates practical applications of Generative AI, prompt engineering, API integration, and structured data validation. It provides a reliable approach to extracting meaningful information from unstructured documents and converting it into machine-readable JSON.
