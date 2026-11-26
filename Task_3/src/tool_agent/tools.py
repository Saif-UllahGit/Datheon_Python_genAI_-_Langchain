"""LangChain tools: real-time web search, safe Python math calculator, and local file reader."""

from __future__ import annotations

import ast
import math
import operator
import os
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Safe AST Python Math Calculator
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_MATH_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}

_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _eval_ast_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown identifier: {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent magnitude exceeds safe limit (1000).")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_ast_node(elt) for elt in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_MATH_FUNCS:
            func_name = getattr(node.func, "id", repr(node.func))
            raise ValueError(f"Function {func_name!r} is not permitted.")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in math expressions.")
        args = [_eval_ast_node(arg) for arg in node.args]
        return _SAFE_MATH_FUNCS[node.func.id](*args)
    raise ValueError(f"Disallowed expression syntax: {type(node).__name__}")


def evaluate_math_expression(expression: str) -> str:
    """Safely evaluate a mathematical expression via Python AST without exec/eval."""
    cleaned = expression.strip()
    if not cleaned:
        return "Error: empty math expression."
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_ast_node(tree)
        if isinstance(result, float) and result.is_integer():
            return f"{cleaned} = {int(result)}"
        if isinstance(result, float):
            return f"{cleaned} = {round(result, 6)}"
        return f"{cleaned} = {result}"
    except Exception as exc:
        return f"Math evaluation error for {cleaned!r}: {exc}"


# ---------------------------------------------------------------------------
# Real-Time Web Search (Tavily optional, DuckDuckGo default)
# ---------------------------------------------------------------------------


def run_web_search(
    query: str,
    max_results: int = 5,
    search_backend: Callable[[str, int], list[dict[str, str]]] | None = None,
) -> str:
    """Search the web via injectable backend, Tavily (if API key set), or DuckDuckGo."""
    cleaned = query.strip()
    if not cleaned:
        return "Error: search query cannot be empty."

    if search_backend is not None:
        results = search_backend(cleaned, max_results)
        return _format_search_results(results, provider_label="CustomSearch")

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            tavily = TavilySearchResults(max_results=max_results)
            raw = tavily.invoke({"query": cleaned})
            normalized = [
                {
                    "title": str(item.get("title", "Result")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", "")),
                }
                for item in (raw if isinstance(raw, list) else [])
            ]
            return _format_search_results(normalized, provider_label="Tavily")
        except Exception:
            pass  # Fall back cleanly to DuckDuckGo

    # Support both newer `ddgs` package and `duckduckgo_search`
    ddgs_cls = None
    try:
        from ddgs import DDGS  # type: ignore

        ddgs_cls = DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore

            ddgs_cls = DDGS
        except ImportError:
            ddgs_cls = None

    if ddgs_cls is not None:
        try:
            with ddgs_cls() as ddgs:
                raw_hits = list(ddgs.text(cleaned, max_results=max_results))
            normalized = [
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("href", item.get("link", ""))),
                    "snippet": str(item.get("body", item.get("snippet", ""))),
                }
                for item in raw_hits
            ]
            return _format_search_results(normalized, provider_label="DuckDuckGo")
        except Exception as exc:
            return f"DuckDuckGo search error: {exc}"

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        return DuckDuckGoSearchRun().run(cleaned)
    except Exception as exc:
        return (
            "Web search unavailable: install `duckduckgo-search` (or `ddgs`) "
            f"or set TAVILY_API_KEY. Details: {exc}"
        )


def _format_search_results(results: list[dict[str, str]], provider_label: str) -> str:
    if not results:
        return f"[{provider_label}] No search results found."
    lines = [f"[{provider_label} Results]"]
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "").strip() or "Untitled"
        url = item.get("url", "").strip()
        snippet = " ".join(item.get("snippet", "").split())
        lines.append(f"{idx}. {title}\n   URL: {url}\n   Snippet: {snippet}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local File Reader
# ---------------------------------------------------------------------------


def read_local_file(
    file_path: str,
    workspace_root: Path | None = None,
    allow_external: bool = False,
    start_line: int = 1,
    end_line: int | None = None,
    max_chars: int = 6000,
) -> str:
    """Safely read a local file (or list directory contents) relative to workspace_root."""
    root = (workspace_root or Path.cwd()).resolve()
    raw = Path(file_path)
    target = (raw if raw.is_absolute() else (root / raw)).resolve()

    if not allow_external:
        try:
            target.relative_to(root)
        except ValueError:
            return (
                f"Access denied: {file_path!r} resolves outside workspace root {root}. "
                "Use a relative path inside the workspace or enable --allow-external."
            )

    if not target.exists():
        return f"File not found: {target}"

    if target.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"Directory listing for {target}:\n" + "\n".join(f"- {e}" for e in entries)

    if target.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(target))
            pages_text = [
                f"[Page {idx}]\n{page.extract_text() or ''}"
                for idx, page in enumerate(reader.pages, start=1)
            ]
            content = "\n\n".join(pages_text)
        except Exception as exc:
            return f"Failed to read PDF {target}: {exc}"
    else:
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Cannot read binary/non-UTF-8 file: {target}"

    lines = content.splitlines()
    start_idx = max(0, start_line - 1)
    stop_idx = end_line if end_line is not None else len(lines)
    sliced = "\n".join(lines[start_idx:stop_idx])

    if len(sliced) > max_chars:
        sliced = sliced[:max_chars] + f"\n... [truncated at {max_chars} chars]"

    return f"File: {target}\nLines: {start_idx + 1}-{min(stop_idx, len(lines))}\n---\n{sliced}"


# ---------------------------------------------------------------------------
# LangChain Tool Schemas & Builder
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for real-time web information.")
    max_results: int = Field(default=5, ge=1, le=10, description="Max search hits to return.")


class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "Pure Python mathematical expression, e.g. '420 * 18500 * 1.145' "
            "or 'sqrt(144) + round(19.456, 2)'."
        ),
    )


class FileReaderInput(BaseModel):
    file_path: str = Field(
        ...,
        description="Path to a local file or directory (relative to workspace root).",
    )
    start_line: int = Field(default=1, ge=1, description="1-based start line to read.")
    end_line: int | None = Field(default=None, ge=1, description="Optional 1-based end line.")


def build_agent_tools(
    workspace_root: Path | None = None,
    allow_external: bool = False,
    search_backend: Callable[[str, int], list[dict[str, str]]] | None = None,
) -> list[BaseTool]:
    """Build the three LangChain tools bound to the agent."""
    root = (workspace_root or Path.cwd()).resolve()

    def _web_search(query: str, max_results: int = 5) -> str:
        return run_web_search(query=query, max_results=max_results, search_backend=search_backend)

    def _calculator(expression: str) -> str:
        return evaluate_math_expression(expression)

    def _file_reader(
        file_path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        return read_local_file(
            file_path=file_path,
            workspace_root=root,
            allow_external=allow_external,
            start_line=start_line,
            end_line=end_line,
        )

    return [
        StructuredTool.from_function(
            func=_web_search,
            name="web_search",
            description=(
                "Search the live web using DuckDuckGo (or Tavily if configured) for current facts, "
                "benchmarks, rates, news, or external knowledge."
            ),
            args_schema=WebSearchInput,
        ),
        StructuredTool.from_function(
            func=_calculator,
            name="python_math_calculator",
            description=(
                "Evaluate mathematical expressions accurately using a safe Python AST calculator. "
                "Always use this tool instead of doing mental arithmetic."
            ),
            args_schema=CalculatorInput,
        ),
        StructuredTool.from_function(
            func=_file_reader,
            name="local_file_reader",
            description=(
                "Read local project files or list workspace directories to inspect metrics, "
                "budgets, configs, or notes."
            ),
            args_schema=FileReaderInput,
        ),
    ]
