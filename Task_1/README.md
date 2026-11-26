# Structured Document Analyzer Using Generative AI and Pydantic

## Introduction

The Structured Document Analyzer is a Python application that extracts important information from unstructured documents, including invoices and resumes, and converts it into structured JSON format.

The application integrates Google Gemini and OpenAI APIs to process document content and identify relevant information. Pydantic v2 is used to validate the extracted data and ensure that the generated output follows predefined schemas.

The system uses prompt engineering, parameterized templates, and few-shot examples to improve information extraction and maintain consistent output.

## Project Objectives

- Extract information from unstructured invoices and resumes.
- Integrate Generative AI models through Gemini and OpenAI APIs.
- Convert document information into structured JSON.
- Implement parameterized prompt templates.
- Use few-shot prompting to improve extraction quality.
- Validate extracted information using Pydantic v2.
- Apply cross-field validation to maintain data consistency.
- Handle invalid responses through validation and correction.
- Support different document types and AI providers.

## Technologies and Tools

| Technology | Application |
|---|---|
| Python 3.10+ | Application development |
| Google Gemini API | Document information extraction |
| OpenAI API | AI-powered document processing |
| Pydantic v2 | Data validation and schema enforcement |
| Prompt Engineering | Improving AI responses |
| Few-Shot Prompting | Guiding information extraction |
| JSON | Structured data representation |
| Unittest | Software testing |

## Project Structure

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

## System Components

### 1. Main Application (analyze.py)

The main Python script controls the execution of the application. It accepts input documents, identifies the document type, selects the AI provider, and processes the extracted information.

### 2. Data Schemas (schemas.py)

Defines Pydantic models for invoices and resumes. These schemas specify the expected fields, data types, and validation rules for the generated information.

### 3. Prompt Templates (prompts.py)

Contains parameterized prompts and few-shot examples designed to guide the AI model in extracting accurate information from different document types.

### 4. Document Analyzer (analyzer.py)

Handles API communication with Google Gemini and OpenAI. It processes AI responses, validates the generated data, and handles validation errors through correction attempts.

### 5. Sample Documents (samples/)

Contains sample invoice and resume documents used to test the document analysis process.

### 6. Test Suite (tests/)

Contains automated tests for data schemas, prompt generation, validation, correction handling, and command-line execution.

## Installation and Configuration

### 1. Prerequisites

- Python 3.10 or later
- pip package manager
- Google Gemini API key or Open
