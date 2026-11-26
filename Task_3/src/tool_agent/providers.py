"""LLM provider factories (Gemini, OpenAI, and offline Mock tool-calling model)."""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

ProviderName = Literal["gemini", "openai", "mock"]

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "mock": "mock-tool-agent-v1",
}


class MockToolCallingChatModel(BaseChatModel):
    """Deterministic offline chat model implementing LangChain bind_tools() for demos & tests."""

    model_name: str = "mock-tool-agent-v1"
    bound_tool_names: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "mock-tool-calling-chat"

    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict[str, Any] | type | Any],
        **kwargs: Any,
    ) -> "MockToolCallingChatModel":
        names = tuple(getattr(t, "name", str(t)) for t in tools)
        return MockToolCallingChatModel(model_name=self.model_name, bound_tool_names=names)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_query = next(
            (str(m.content) for m in messages if isinstance(m, HumanMessage)),
            "",
        )
        tool_observations = [m for m in messages if isinstance(m, ToolMessage)]
        called_tools = {m.name for m in tool_observations if m.name}

        # 1. Inspect if query references a local file and we haven't read it yet.
        file_match = re.search(r"([A-Za-z0-9_./\\-]+\.(?:txt|json|csv|md|py|pdf))", user_query)
        if (
            file_match
            and "local_file_reader" in self.bound_tool_names
            and "local_file_reader" not in called_tools
        ):
            target_file = file_match.group(1)
            msg = AIMessage(
                content=f"I need to read `{target_file}` first to gather the local figures.",
                tool_calls=[
                    {
                        "name": "local_file_reader",
                        "args": {"file_path": target_file},
                        "id": "call_mock_file_1",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # 2. Check if calculation is requested and hasn't been executed yet.
        needs_math = bool(
            re.search(r"\b(calculate|compute|multiply|sum|total|arr|growth|projected|budget)\b", user_query, re.I)
            or re.search(r"\d+\s*[+\-*/]\s*\d+", user_query)
        )
        if (
            needs_math
            and "python_math_calculator" in self.bound_tool_names
            and "python_math_calculator" not in called_tools
        ):
            expr = self._derive_math_expression(user_query, tool_observations)
            msg = AIMessage(
                content=f"Now I will calculate `{expr}` using the Python math calculator tool.",
                tool_calls=[
                    {
                        "name": "python_math_calculator",
                        "args": {"expression": expr},
                        "id": "call_mock_calc_2",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # 3. Check if web search is requested and hasn't been executed yet.
        needs_web = bool(
            re.search(r"\b(search|web|online|current|latest|benchmark|news|rate|duckduckgo)\b", user_query, re.I)
        )
        if (
            needs_web
            and "web_search" in self.bound_tool_names
            and "web_search" not in called_tools
        ):
            msg = AIMessage(
                content="Next, I will run a real-time web search to retrieve external context.",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": user_query, "max_results": 3},
                        "id": "call_mock_web_3",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # 4. All required tools called -> synthesize Final Answer from observations.
        if not tool_observations:
            summary = f"Direct response (no tools needed): {user_query}"
        else:
            obs_lines = [f"- [{m.name}] {str(m.content).strip()}" for m in tool_observations]
            summary = "Synthesized from tool observations:\n" + "\n".join(obs_lines)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=summary))])

    @staticmethod
    def _derive_math_expression(query: str, observations: list[ToolMessage]) -> str:
        explicit = re.search(r"([0-9]+(?:\.[0-9]+)?(?:\s*[+\-*/]\s*[0-9]+(?:\.[0-9]+)?)+)", query)
        if explicit:
            return explicit.group(1).strip()

        obs_text = "\n".join(str(m.content) for m in observations)
        cust = re.search(r"Active Enterprise Customers:\s*(\d+)", obs_text)
        acv = re.search(r"Average Annual Contract Value \(ACV\):\s*\$([0-9,]+)", obs_text)
        growth = re.search(r"Target YoY Revenue Growth Rate:\s*([0-9.]+)%", obs_text)
        if cust and acv:
            c_val = cust.group(1)
            a_val = acv.group(1).replace(",", "")
            if growth and "growth" in query.lower():
                g_val = float(growth.group(1)) / 100.0
                return f"{c_val} * {a_val} * {1 + g_val}"
            return f"{c_val} * {a_val}"
        return "2 + 2"


def create_chat_model(
    provider: ProviderName = "gemini",
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Instantiate a LangChain chat model supporting bind_tools()."""
    if provider == "mock":
        return MockToolCallingChatModel()

    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini provider.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"],
            google_api_key=key,
            temperature=temperature,
        )

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OPENAI_API_KEY for OpenAI provider.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL") or DEFAULT_MODELS["openai"],
            api_key=key,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
