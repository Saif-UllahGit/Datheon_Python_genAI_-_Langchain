"""LangChain tool-calling research agent with web search, math calculator, and file reader."""

from .executor import AgentStep, AgentTrace, ConsoleAgentExecutor
from .providers import create_chat_model
from .tools import build_agent_tools, evaluate_math_expression, read_local_file, run_web_search

__all__ = [
    "AgentStep",
    "AgentTrace",
    "ConsoleAgentExecutor",
    "build_agent_tools",
    "create_chat_model",
    "evaluate_math_expression",
    "read_local_file",
    "run_web_search",
]
