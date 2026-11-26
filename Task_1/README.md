# Structured Document Analyzer
### Gemini / OpenAI API Integration with Pydantic v2

## 1. Project Overview

The Structured Document Analyzer is a Python-based application designed to extract meaningful information from unstructured documents, particularly invoices and resumes, and convert it into structured, validated JSON data.

The application integrates Google Gemini and OpenAI language models to analyze document content and identify relevant information. It uses parameterized few-shot prompt templates to guide the extraction process and Pydantic v2 to validate the generated responses against predefined data schemas.

The system also incorporates validation and correction mechanisms to improve the reliability and consistency of the extracted information.

## 2. Project Objectives

- Extract relevant information from unstructured invoices and resumes.
- Integrate Google Gemini and OpenAI APIs for document analysis.
- Convert extracted information into structured JSON.
- Implement parameterized prompt templates.
- Apply few-shot prompting to improve extraction accuracy.
- Validate generated responses using Pydantic v2.
- Implement cross-field validation rules.
- Handle invalid or incomplete model responses.
- Support multiple AI providers and offline testing.

## 3. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core application development |
| Google Gemini API | AI-powered document analysis |
| OpenAI API | Alternative language model integration |
| Pydantic v2 | Structured data validation |
| Prompt Engineering | Improving information extraction |
| Few-Shot Prompting | Guiding model responses with examples |
| JSON | Structured output representation |
| Python Unittest | Automated testing |

## 4. Project Structure

```text
Task_1/
│
├── analyze.py
├── requirements.txt
├── .env.example
│
├── samples/
│   ├── messy_invoice.txt
│   └── messy_resume.txt
│
├── src/
│   └── doc_analyzer/
│       ├── __init__.py
│       ├── schemas.py
│       ├── prompts.py
│       └── analyzer.py
│
└── tests/
    └── test_pipeline.py
```

## 5. System Components

### 5.1 Main Application (analyze.py)

Provides the command-line interface for executing document analysis. It accepts input documents, identifies the document type, selects the AI provider, and generates structured output.

### 5.2 Data Schemas (schemas.py)

Defines Pydantic v2 models for invoices and resumes. These models specify the expected fields, data types, constraints, and validation rules for extracted information.

### 5.3 Prompt Templates (prompts.py)

Contains parameterized prompt templates and few-shot examples that guide the language model in extracting relevant information from different document types.

### 5.4 Document Analyzer (analyzer.py)

Manages communication with Gemini and OpenAI APIs. It processes model responses, validates the generated JSON, and handles validation errors through correction attempts.

### 5.5 Sample Documents (samples/)

Contains sample invoice and resume text files used to demonstrate and evaluate the document extraction process.

### 5.6 Automated Tests (tests/)

Contains test cases for schema validation, prompt generation, retry mechanisms, and command-line execution.

## 6. Installation and Setup

### 6.1 Prerequisites

- Python 3.10 or later
- pip package manager
- Google Gemini API key or OpenAI API key

### 6.2 Create a Virtual Environment

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

### 6.3 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 6.4 Configure API Keys

Create a `.env` file using `.env.example` as a template.

For Google Gemini:

```env
GEMINI_API_KEY=your_gemini_api_key
```

For OpenAI:

```env
OPENAI_API_KEY=your_openai_api_key
```

Ensure that API keys remain private and that the `.env` file is excluded from version control.

## 7. Application Usage

### 7.1 Analyze an Invoice Using Gemini

```bash
python analyze.py --type invoice --input samples/messy_invoice.txt --provider gemini --currency USD
```

This command processes an invoice using Google Gemini and extracts the relevant invoice information into structured JSON.

### 7.2 Analyze a Resume Using OpenAI

```bash
python analyze.py --type resume --input samples/messy_resume.txt --provider openai --target-role "Staff ML Platform Engineer" --output resume_out.json
```

This command analyzes a resume, extracts professional information, and evaluates its relevance to the specified target role.

### 7.3 Inspect the Generated Prompt

```bash
python analyze.py --type resume --input samples/messy_resume.txt --target-role "Backend Engineer" --show-prompt
```

This command displays the parameterized prompt before sending it to the selected language model.

### 7.4 Run Offline Using Mock Provider

```bash
python analyze.py --type invoice --input samples/messy_invoice.txt --provider mock
```

This option allows the application to be tested without requiring an external AI API, provided the mock provider is implemented.

## 8. Output and Validation

The application generates structured JSON responses according to the predefined Pydantic schemas.

The validation process checks:

- Required fields and data types.
- Correct data structure.
- Field-level constraints.
- Cross-field consistency.
- Valid JSON responses.

If the generated response fails validation, the application can attempt to correct it through the configured retry mechanism.

## 9. Running Automated Tests

Execute the following command to run the test suite:

```bash
python -m unittest discover -s tests -v
```

The test suite covers:

1. Invoice schema validation.
2. Resume schema validation.
3. Parameterized prompt generation.
4. Few-shot example integration.
5. Invalid response handling.
6. Validation and correction attempts.
7. Command-line functionality.

## 10. Key Features

- Automated invoice and resume information extraction.
- Google Gemini and OpenAI API integration.
- Parameterized few-shot prompt templates.
- Structured JSON generation.
- Pydantic v2 schema validation.
- Cross-field validation.
- Automatic validation and correction attempts.
- Offline mock provider support.
- Command-line interface.
- Automated testing.

## 11. Expected Deliverable

A functional Python application capable of processing unstructured invoice and resume documents and converting their content into structured, validated JSON using Google Gemini or OpenAI.

The application demonstrates practical implementation of Generative AI, API integration, prompt engineering, few-shot learning, and structured data validation.

## 12. Conclusion

The Structured Document Analyzer demonstrates how Generative AI can be applied to real-world document processing tasks. By combining Python, language model APIs, prompt engineering, and Pydantic v2, the system provides an organized approach to extracting information from unstructured documents and transforming it into reliable, machine-readable JSON data.
