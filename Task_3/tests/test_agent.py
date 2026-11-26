"""Unit and end-to-end tests for tools, bind_tools loop, and console CLI."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent_cli  # noqa: E402
from tool_agent.executor import ConsoleAgentExecutor  # noqa: E402
from tool_agent.providers import MockToolCallingChatModel  # noqa: E402
from tool_agent.tools import (  # noqa: E402
    build_agent_tools,
    evaluate_math_expression,
    read_local_file,
    run_web_search,
)


class CalculatorToolTests(unittest.TestCase):
    def test_evaluates_math_functions_and_arithmetic(self) -> None:
        self.assertEqual(evaluate_math_expression("420 * 18500 * 1.145"), "420 * 18500 * 1.145 = 8896650")
        self.assertEqual(evaluate_math_expression("sqrt(144) + round(pi, 2)"), "sqrt(144) + round(pi, 2) = 15.14")

    def test_blocks_unsafe_code_execution(self) -> None:
        blocked = evaluate_math_expression("__import__('os').system('echo pwned')")
        self.assertIn("error", blocked.lower())


class LocalFileAndSearchTests(unittest.TestCase):
    def test_reads_workspace_file_and_blocks_traversal(self) -> None:
        ok = read_local_file("samples/company_metrics.txt", workspace_root=PROJECT_ROOT)
        self.assertIn("Active Enterprise Customers: 420", ok)

        denied = read_local_file("../../outside.txt", workspace_root=PROJECT_ROOT)
        self.assertIn("Access denied", denied)

    def test_web_search_formats_injected_backend(self) -> None:
        def fake_search(query: str, max_results: int) -> list[dict[str, str]]:
            return [
                {
                    "title": "SaaS Benchmarks 2026",
                    "url": "https://example.com/saas",
                    "snippet": "Median enterprise SaaS NRR is 112%.",
                }
            ]

        output = run_web_search("SaaS NRR benchmark", max_results=2, search_backend=fake_search)
        self.assertIn("SaaS Benchmarks 2026", output)
        self.assertIn("112%", output)


class AgentExecutorTests(unittest.TestCase):
    def test_compound_query_invokes_file_calculator_and_web_search(self) -> None:
        def fake_search(query: str, max_results: int) -> list[dict[str, str]]:
            return [
                {
                    "title": "SaaS Growth Benchmarks",
                    "url": "https://example.com/benchmarks",
                    "snippet": "Top-quartile SaaS ARR growth is 15% YoY.",
                }
            ]

        tools = build_agent_tools(workspace_root=PROJECT_ROOT, search_backend=fake_search)
        llm = MockToolCallingChatModel()
        log_buf = io.StringIO()
        executor = ConsoleAgentExecutor(llm=llm, tools=tools, verbose=True, stream=log_buf)

        trace = executor.run(
            "Read samples/company_metrics.txt, calculate projected ARR with growth, "
            "and search the web for current SaaS ARR growth benchmarks."
        )

        called_tools = [step.action_tool for step in trace.steps]
        self.assertEqual(
            called_tools,
            ["local_file_reader", "python_math_calculator", "web_search"],
        )
        self.assertIn("8896650", trace.final_answer)
        rendered_log = log_buf.getvalue()
        self.assertIn("Thought:", rendered_log)
        self.assertIn("Action:", rendered_log)
        self.assertIn("Observation:", rendered_log)
        self.assertIn("Final Answer:", rendered_log)

    def test_cli_json_mode_runs_end_to_end(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = agent_cli.main(
                [
                    "--provider",
                    "mock",
                    "--query",
                    "Read samples/project_budget.json and calculate 3200 + 4100 + 5200",
                    "--json",
                ]
            )
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual([s["action"] for s in payload["steps"]], ["local_file_reader", "python_math_calculator"])
        self.assertIn("12500", payload["final_answer"])


if __name__ == "__main__":
    unittest.main()
