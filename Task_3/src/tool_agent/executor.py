"""Agent executor loop built on LangChain bind_tools() displaying Thought -> Action -> Observation -> Final Answer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from typing import Any, Sequence, TextIO

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

AGENT_SYSTEM_PROMPT = """You are an autonomous research and problem-solving agent.
You have access to three tools:
1. web_search: Search the live web for current facts, rates, benchmarks, or external data.
2. python_math_calculator: Evaluate mathematical formulas accurately using Python AST.
3. local_file_reader: Read local workspace files or inspect workspace directories.

OPERATING GUIDELINES:
- For compound queries, decompose the problem and call tools step-by-step until you have verified every fact.
- Before calling any tool, briefly state your reasoning (Thought) explaining which tool you are choosing and why.
- Never do non-trivial arithmetic in your head; always call python_math_calculator.
- Never fabricate local file contents or live web facts; cite observations directly in your Final Answer."""


@dataclass(frozen=True)
class AgentStep:
    """Single Thought -> Action -> Observation record from the executor loop."""

    step_number: int
    thought: str
    action_tool: str
    action_input: dict[str, Any]
    observation: str


@dataclass(frozen=True)
class AgentTrace:
    """Complete execution trace and final answer for a user query."""

    query: str
    steps: tuple[AgentStep, ...]
    final_answer: str
    iterations: int


def _extract_text_content(content: Any) -> str:
    """Normalize string or multimodal block lists from LangChain AIMessage.content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(content).strip() if content is not None else ""


class ConsoleAgentExecutor:
    """Autonomous tool-calling agent loop displaying Thought -> Action -> Observation -> Final Answer."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        max_iterations: int = 8,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        verbose: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.bound_llm = llm.bind_tools(list(tools))
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.stream = stream or sys.stdout

    def _log(self, line: str) -> None:
        if self.verbose:
            print(line, file=self.stream, flush=True)

    def run(self, query: str) -> AgentTrace:
        """Execute the autonomous tool-calling loop for a compound user query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must be a non-empty string.")

        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cleaned_query),
        ]
        recorded_steps: list[AgentStep] = []
        step_counter = 0

        for iteration in range(1, self.max_iterations + 1):
            ai_message = self.bound_llm.invoke(messages)
            if not isinstance(ai_message, AIMessage):
                ai_message = AIMessage(content=_extract_text_content(ai_message))

            messages.append(ai_message)
            thought_text = _extract_text_content(ai_message.content)
            tool_calls = getattr(ai_message, "tool_calls", None) or []

            if not tool_calls:
                final_answer = thought_text or "No answer generated."
                self._log(f"\nFinal Answer: {final_answer}")
                return AgentTrace(
                    query=cleaned_query,
                    steps=tuple(recorded_steps),
                    final_answer=final_answer,
                    iterations=iteration,
                )

            for call in tool_calls:
                step_counter += 1
                tool_name = str(call.get("name", ""))
                tool_args = dict(call.get("args", {}) or {})
                call_id = str(call.get("id") or f"call_{step_counter}")

                display_thought = (
                    thought_text
                    if thought_text
                    else f"Calling `{tool_name}` to gather required information."
                )
                self._log(f"\n[Step {step_counter}] Thought: {display_thought}")
                self._log(f"[Step {step_counter}] Action: {tool_name}({json.dumps(tool_args)})")

                observation = self._invoke_tool(tool_name, tool_args)
                self._log(f"[Step {step_counter}] Observation: {observation}")

                recorded_steps.append(
                    AgentStep(
                        step_number=step_counter,
                        thought=display_thought,
                        action_tool=tool_name,
                        action_input=tool_args,
                        observation=observation,
                    )
                )
                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )

        fallback = (
            f"Stopped after reaching max_iterations ({self.max_iterations}). "
            "Try breaking the query into smaller steps or increasing --max-iterations."
        )
        self._log(f"\nFinal Answer: {fallback}")
        return AgentTrace(
            query=cleaned_query,
            steps=tuple(recorded_steps),
            final_answer=fallback,
            iterations=self.max_iterations,
        )

    def _invoke_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self.tools))
            return f"Error: unknown tool {tool_name!r}. Available tools: {available}"
        try:
            result = tool.invoke(tool_args)
            return str(result)
        except Exception as exc:
            return f"Tool execution error in {tool_name}: {exc}"
