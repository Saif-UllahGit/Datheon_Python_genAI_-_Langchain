#!/usr/bin/env python3
"""Console AI agent that autonomously selects and invokes tools for compound queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from tool_agent.executor import ConsoleAgentExecutor
from tool_agent.providers import create_chat_model
from tool_agent.tools import build_agent_tools


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Console LangChain tool-calling agent with DuckDuckGo/Tavily web search, "
            "Python AST math calculator, and local file reader."
        )
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Single compound research query to execute.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive console session.",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "mock"],
        help="LLM provider (default: gemini).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name override.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT,
        help="Root directory accessible by local_file_reader (default: project root).",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow local_file_reader to access files outside --workspace.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum reasoning/tool iterations per query (default: 8).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full reasoning trace and final answer as JSON.",
    )
    return parser.parse_args(argv)


def _print_trace_json(trace) -> None:
    payload = {
        "query": trace.query,
        "iterations": trace.iterations,
        "steps": [
            {
                "step": s.step_number,
                "thought": s.thought,
                "action": s.action_tool,
                "action_input": s.action_input,
                "observation": s.observation,
            }
            for s in trace.steps
        ],
        "final_answer": trace.final_answer,
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.query and not args.interactive:
        print("Error: pass --query '...' or --interactive.", file=sys.stderr)
        return 1

    try:
        llm = create_chat_model(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            temperature=0.0,
        )
        tools = build_agent_tools(
            workspace_root=args.workspace,
            allow_external=args.allow_external,
        )
        executor = ConsoleAgentExecutor(
            llm=llm,
            tools=tools,
            max_iterations=args.max_iterations,
            verbose=not args.json,
        )

        if args.query:
            trace = executor.run(args.query)
            if args.json:
                _print_trace_json(trace)

        if args.interactive:
            print(
                f"Interactive Agent Ready (provider={args.provider}, workspace={args.workspace.resolve()}). "
                "Type 'exit' or 'quit' to stop.",
                file=sys.stderr,
            )
            while True:
                try:
                    user_q = input("\nQuery> ").strip()
                except EOFError:
                    break
                if not user_q or user_q.lower() in {"exit", "quit"}:
                    break
                trace = executor.run(user_q)
                if args.json:
                    _print_trace_json(trace)
                else:
                    print("\n" + "=" * 60)

    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
