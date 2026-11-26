# Structured Document Analyzer (Gemini / OpenAI + Pydantic)

Executable Python pipeline for parsing noisy, unstructured **invoices** and **resumes** into strictly validated JSON schemas using parameterized few-shot prompt templates, **Google Gemini** (`google-generativeai`) or **OpenAI** (`openai`), and **Pydantic v2**.

## Project Structure

```text
structured-doc-analyzer/
├── analyze.py                  # Executable CLI entrypoint
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── samples/
│   ├── messy_invoice.txt       # Sample noisy invoice document
│   └── messy_resume.txt        # Sample unstructured resume document
├── src/doc_analyzer/
│   ├── __init__.py
│   ├── schemas.py              # Strict Pydantic v2 schemas & cross-field validators
│   ├── prompts.py              # Parameterized prompt templates & few-shot examples
│   └── analyzer.py             # Gemini/OpenAI/Mock clients + auto-repair validation loop
└── tests/
    └── test_pipeline.py        # Schema, prompt, retry-loop, and CLI tests
```

## Setup

1. Ensure Python 3.10+ is installed, then create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

2. Configure an API key (copy `.env.example` to `.env` or export in your shell):

```bash
# For Google Gemini
export GEMINI_API_KEY="your-key"

# Or for OpenAI
export OPENAI_API_KEY="your-key"
```

## Usage

### Analyze an Invoice (Gemini)

```bash
python analyze.py --type invoice --input samples/messy_invoice.txt --provider gemini --currency USD
```

### Analyze a Resume with Target Role Fit (OpenAI)

```bash
python analyze.py --type resume --input samples/messy_resume.txt --provider openai --target-role "Staff ML Platform Engineer" --output resume_out.json
```

### Inspect the Rendered Parameterized Prompt

```bash
python analyze.py --type resume --input samples/messy_resume.txt --target-role "Backend Engineer" --show-prompt
```

### Run Offline with Mock Provider

```bash
python analyze.py --type invoice --input samples/messy_invoice.txt --provider mock
```

## Running Tests

```bash
python -m unittest discover -s tests -v
```
