## LangChain Autonomous Tool-Calling Research Agent

## Project Overview

The LangChain Autonomous Tool-Calling Research Agent is a Python-based console application that uses Large Language Models (LLMs) to perform research, calculations, and local file analysis through external tools.

The agent is developed using LangChain tool definitions (`StructuredTool`), `llm.bind_tools()`, and an execution loop that follows the Thought → Action → Observation → Final Answer approach.

It can understand user instructions, select appropriate tools, execute actions, analyze tool results, and generate final responses based on the collected information.

## Objectives

- Develop an autonomous AI agent using LangChain.
- Implement structured tool definitions using `StructuredTool`.
- Integrate LLM tool calling through `llm.bind_tools()`.
- Implement a Thought → Action → Observation → Final Answer execution loop.
- Perform live web searches.
- Execute mathematical calculations safely.
- Read and analyze local files.
- Support Gemini and OpenAI language models.
- Provide an offline mock mode for testing.

## Equipped Tools

### 1. Web Search (`web_search`)

Performs web searches to retrieve relevant information from online sources.

- Uses DuckDuckGo through `duckduckgo-search` or `ddgs`.
- Supports Tavily search when `TAVILY_API_KEY` is configured.
- Provides information to the agent for research-based queries.

### 2. Python Math Calculator (`python_math_calculator`)

Performs mathematical calculations using a restricted Python AST evaluator.

Supported operations include:

- Basic arithmetic operations.
- Square roots using `sqrt`.
- Logarithmic calculations using `log`.
- Trigonometric functions.
- Rounding using `round`.
- Minimum and maximum calculations.
- Mathematical constants such as `pi`, `e`, and `tau`.

The calculator avoids unsafe use of Python's `eval()` function.

### 3. Local File Reader (`local_file_reader`)

Reads and analyzes files from an authorized local workspace.

Supported file formats:

- TXT
- JSON
- CSV
- Markdown
- Python
- PDF

Additional functionality includes directory reading, line-based content selection, and path-traversal protection.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Main programming language |
| LangChain | AI agent development |
| StructuredTool | Structured tool definitions |
| Gemini API | Language model integration |
| OpenAI API | Alternative language model integration |
| DuckDuckGo | Web search |
| Tavily | Alternative web search provider |
| Python AST | Safe mathematical evaluation |
| PyPDF | PDF file reading |
| Unittest | Automated testing |

## Project Structure

```text
Task_3/
│
├── agent_cli.py
├── requirements.txt
├── .env.example
│
├── samples/
│   ├── company_metrics.txt
│   └── project_budget.json
│
├── src/
│   └── tool_agent/
│       ├── __init__.py
│       ├── tools.py
│       ├── providers.py
│       └── executor.py
│
└── tests/
    └── test_agent.py
```

## System Components

### 1. Agent CLI (`agent_cli.py`)

Provides the command-line interface for interacting with the AI agent. It accepts user queries, selects the language model provider, and supports both single-query and interactive execution modes.

### 2. Tool Definitions (`tools.py`)

Defines the available tools for web searching, mathematical calculations, and local file reading. Each tool provides a specific capability that the agent can invoke when required.

### 3. Model Providers (`providers.py`)

Handles integration with Google Gemini and OpenAI models. It also provides a mock model for offline testing without requiring an external API connection.

### 4. Agent Executor (`executor.py`)

Controls the agent's execution process. It manages tool selection, tool calls, observations, and final response generation through an explicit execution loop.

### 5. Sample Files (`samples/`)

Contains sample text and JSON files used to demonstrate file-reading and calculation capabilities.

### 6. Automated Tests (`tests/`)

Contains unit and end-to-end tests for tool execution, mathematical calculations, file reading, and agent behavior.

## Installation and Setup

### 1. Prerequisites

- Python 3.10 or later
- pip package manager
- Google Gemini API key or OpenAI API key
- Internet connection for live web searches

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate the environment on Windows:

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

### 4. Configure Environment Variables

Create a `.env` file using `.env.example` as a template.

For Google Gemini:

```env
GEMINI_API_KEY=your_gemini_api_key
```

For OpenAI:

```env
OPENAI_API_KEY=your_openai_api_key
```

For Tavily search (optional):

```env
TAVILY_API_KEY=your_tavily_api_key
```

Keep API keys private and never upload the `.env` file to GitHub.

## Usage

### 1. Run a Research Query Using Gemini

```powershell
python agent_cli.py --provider gemini -q "Read samples/company_metrics.txt, calculate projected ARR after the target YoY growth rate, and search DuckDuckGo for 2026 median SaaS growth benchmarks."
```

This query demonstrates the agent's ability to read a local file, perform mathematical calculations, and retrieve information through web search.

### 2. Interactive Console Mode Using OpenAI

```powershell
python agent_cli.py --provider openai --interactive
```

This starts an interactive session where users can submit multiple queries.

### 3. Run Offline Using the Mock Provider

```powershell
python agent_cli.py --provider mock -q "Read samples/project_budget.json and calculate 3200 + 4100 + 5200"
```

The mock provider allows the application to demonstrate its supported offline functionality without requiring an external AI API.

## Running Tests

Execute the following command to run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

1. Web search tool functionality.
2. Mathematical expression evaluation.
3. Local file reading.
4. File access restrictions.
5. Tool execution and observations.
6. Agent execution flow.
7. Offline mock provider behavior.

## Key Features

- Autonomous tool selection and execution.
- LangChain `StructuredTool` integration.
- LLM tool calling using `bind_tools()`.
- Explicit Thought → Action → Observation → Final Answer execution flow.
- Live web research using DuckDuckGo and optional Tavily.
- Safe mathematical calculations using Python AST.
- Local TXT, JSON, CSV, Markdown, Python, and PDF file reading.
- Gemini and OpenAI model support.
- Offline mock provider.
- Interactive command-line interface.
- Automated unit and end-to-end testing.

## Expected Deliverable

A functional Python-based autonomous research agent capable of understanding user queries, selecting appropriate tools, performing web searches, executing mathematical calculations, reading local files, and generating responses based on tool observations.

## Conclusion

This project demonstrates the practical implementation of autonomous AI agents using Python and LangChain. By combining LLM tool calling, web search, mathematical evaluation, and local file processing, the system provides a flexible approach to building AI agents capable of performing multiple tasks through structured tool execution.