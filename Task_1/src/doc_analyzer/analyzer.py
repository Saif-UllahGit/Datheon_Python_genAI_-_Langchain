"""LLM client adapters, JSON extraction, and Pydantic validation with repair retries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Literal, Type

from pydantic import ValidationError

from .prompts import PromptParameters, RenderedPrompt, build_prompts
from .schemas import DocumentType, StrictModel, get_schema_for_type

ProviderName = Literal["gemini", "openai", "mock"]

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "mock": "mock-deterministic-v1",
}


class AnalysisError(RuntimeError):
    """Raised when document extraction or strict Pydantic validation fails."""


@dataclass(frozen=True)
class AnalysisResult:
    """Validated structured output alongside execution metadata."""

    document_type: DocumentType
    provider: ProviderName
    model: str
    attempts: int
    data: StrictModel
    raw_json: str


def strip_markdown_json_fences(raw_text: str) -> str:
    """Remove optional ```json ... ``` wrappers or surrounding non-JSON noise."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def _build_repair_prompt(bad_output: str, error_details: str) -> str:
    return (
        "Your previous JSON output failed strict schema validation.\n"
        f"VALIDATION ERROR:\n{error_details}\n\n"
        f"INVALID OUTPUT:\n{bad_output}\n\n"
        "Return ONLY a corrected JSON object that satisfies every schema constraint."
    )


class BaseLLMClient:
    """Minimal protocol for provider adapters."""

    def generate_json(
        self,
        rendered_prompt: RenderedPrompt,
        schema_cls: Type[StrictModel],
        repair_turns: list[tuple[str, str]] | None = None,
    ) -> str:
        raise NotImplementedError


class GeminiClient(BaseLLMClient):
    """Google Gemini adapter using google-generativeai with JSON MIME enforcement."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODELS["gemini"],
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise AnalysisError(
                "Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key."
            )
        self.model = model
        self.temperature = temperature

        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise AnalysisError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self.api_key)
        self._genai = genai

    def generate_json(
        self,
        rendered_prompt: RenderedPrompt,
        schema_cls: Type[StrictModel],
        repair_turns: list[tuple[str, str]] | None = None,
    ) -> str:
        model_instance = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=rendered_prompt.system_instruction,
        )
        prompt_parts: list[str] = [rendered_prompt.as_single_user_prompt()]
        if repair_turns:
            for bad_output, error_details in repair_turns:
                prompt_parts.append(_build_repair_prompt(bad_output, error_details))

        generation_config = self._genai.GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        response = model_instance.generate_content(
            "\n\n".join(prompt_parts),
            generation_config=generation_config,
        )
        if not getattr(response, "text", None):
            raise AnalysisError("Gemini returned an empty response.")
        return response.text


class OpenAIClient(BaseLLMClient):
    """OpenAI adapter using structured JSON output and Pydantic parsing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODELS["openai"],
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AnalysisError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key."
            )
        self.model = model
        self.temperature = temperature

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise AnalysisError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=self.api_key)

    def generate_json(
        self,
        rendered_prompt: RenderedPrompt,
        schema_cls: Type[StrictModel],
        repair_turns: list[tuple[str, str]] | None = None,
    ) -> str:
        messages = rendered_prompt.as_openai_messages()
        if repair_turns:
            for bad_output, error_details in repair_turns:
                messages.append({"role": "assistant", "content": bad_output})
                messages.append(
                    {"role": "user", "content": _build_repair_prompt(bad_output, error_details)}
                )

        # Prefer OpenAI's native Pydantic structured output parser when available.
        beta = getattr(self._client, "beta", None)
        chat_beta = getattr(beta, "chat", None) if beta else None
        completions_beta = getattr(chat_beta, "completions", None) if chat_beta else None
        if completions_beta and hasattr(completions_beta, "parse"):
            parsed_resp = completions_beta.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=schema_cls,
            )
            message = parsed_resp.choices[0].message
            if getattr(message, "parsed", None) is not None:
                return message.parsed.model_dump_json()
            if getattr(message, "content", None):
                return message.content

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise AnalysisError("OpenAI returned an empty response.")
        return content


class MockClient(BaseLLMClient):
    """Offline deterministic adapter for testing prompt & Pydantic validation locally."""

    def __init__(self, target_role: str | None = None) -> None:
        self.target_role = target_role

    def generate_json(
        self,
        rendered_prompt: RenderedPrompt,
        schema_cls: Type[StrictModel],
        repair_turns: list[tuple[str, str]] | None = None,
    ) -> str:
        text = rendered_prompt.user_prompt
        if schema_cls.__name__ == "InvoiceAnalysis":
            payload = self._mock_invoice(text)
        else:
            payload = self._mock_resume(text)
        return json.dumps(payload, indent=2)

    def _mock_invoice(self, text: str) -> dict[str, Any]:
        inv_match = re.search(r"INV(?:OICE)?\s*#?\s*([A-Z0-9_-]+)", text, re.IGNORECASE)
        invoice_number = inv_match.group(1) if inv_match else "MOCK-INV-001"
        return {
            "invoice_number": invoice_number,
            "issue_date": "2026-11-10",
            "due_date": "2026-12-10",
            "currency": "USD",
            "vendor": {
                "name": "Sample Vendor Corp",
                "tax_id": "US-12-3456789",
                "email": "billing@samplevendor.example",
                "phone": "+1-415-555-0199",
                "address": "100 Main St, San Francisco, CA",
            },
            "customer": {
                "name": "Sample Client Ltd",
                "tax_id": None,
                "email": "accounts@sampleclient.example",
                "phone": None,
                "address": "200 Oak Ave, Austin, TX",
            },
            "line_items": [
                {
                    "description": "Extracted Consulting Service",
                    "quantity": 10.0,
                    "unit_price": 150.0,
                    "tax_rate_percent": 10.0,
                    "line_total": 1500.0,
                }
            ],
            "subtotal": 1500.0,
            "tax_amount": 150.0,
            "discount_amount": 0.0,
            "total_amount": 1650.0,
            "payment_terms": "Net 30",
            "payment_status": "unpaid",
            "confidence_notes": ["Generated by offline MockClient for pipeline verification."],
        }

    def _mock_resume(self, text: str) -> dict[str, Any]:
        role_match = re.search(r"Target role for fit evaluation:\s*(.+)", text)
        role_name = role_match.group(1).strip() if role_match else self.target_role
        assessment = (
            {
                "target_role": role_name,
                "match_score": 86,
                "strengths": ["Strong Python backend and cloud architecture experience."],
                "gaps": ["Verify domain-specific scale metrics during interview."],
            }
            if role_name
            else None
        )
        return {
            "candidate_name": "Alex Rivera",
            "contact": {
                "email": "alex.rivera@example.com",
                "phone": "+1-555-0142",
                "location": "New York, NY",
                "linkedin_url": "https://linkedin.com/in/alexrivera",
                "github_url": "https://github.com/arivera",
                "portfolio_url": None,
            },
            "professional_summary": "Senior Python and ML platform engineer with 6 years of experience.",
            "years_of_experience": 6.0,
            "overall_seniority": "senior",
            "skills": {
                "programming_languages": ["Python", "SQL", "TypeScript"],
                "frameworks_and_tools": ["FastAPI", "Pydantic", "Docker", "Kubernetes"],
                "domain_skills": ["LLM Integration", "Distributed Systems"],
                "spoken_languages": ["English"],
            },
            "work_experience": [
                {
                    "company": "Vertex Data Labs",
                    "role": "Senior Backend Engineer",
                    "start_date": "2022-04",
                    "end_date": None,
                    "is_current": True,
                    "location": "New York, NY",
                    "highlights": [
                        "Designed structured LLM document extraction pipelines with Pydantic validation."
                    ],
                }
            ],
            "education": [
                {
                    "institution": "Cornell University",
                    "degree": "B.S.",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2020,
                    "gpa": "3.7",
                }
            ],
            "certifications": ["Google Cloud Professional Data Engineer"],
            "projects": [],
            "role_fit_assessment": assessment,
            "confidence_notes": ["Generated by offline MockClient for pipeline verification."],
        }


class DocumentAnalyzer:
    """Orchestrates parameterized prompting, LLM invocation, and strict Pydantic validation."""

    def __init__(
        self,
        provider: ProviderName = "gemini",
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        client: BaseLLMClient | None = None,
    ) -> None:
        self.provider: ProviderName = provider
        self.model = model or os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
        self.max_retries = max(0, max_retries)
        self._client = client or self._create_client(provider, self.model, api_key, temperature)

    @staticmethod
    def _create_client(
        provider: ProviderName,
        model: str,
        api_key: str | None,
        temperature: float,
    ) -> BaseLLMClient:
        if provider == "gemini":
            return GeminiClient(api_key=api_key, model=model, temperature=temperature)
        if provider == "openai":
            return OpenAIClient(api_key=api_key, model=model, temperature=temperature)
        if provider == "mock":
            return MockClient()
        raise AnalysisError(f"Unsupported provider: {provider!r}")

    def analyze(self, document_text: str, params: PromptParameters) -> AnalysisResult:
        """Parse unstructured text and validate against the target Pydantic schema."""
        schema_cls = get_schema_for_type(params.document_type)
        rendered = build_prompts(document_text, params)
        repair_turns: list[tuple[str, str]] = []
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 2):
            raw_response = self._client.generate_json(
                rendered_prompt=rendered,
                schema_cls=schema_cls,
                repair_turns=repair_turns or None,
            )
            cleaned_json = strip_markdown_json_fences(raw_response)

            try:
                validated = schema_cls.model_validate_json(cleaned_json)
                return AnalysisResult(
                    document_type=params.document_type,
                    provider=self.provider,
                    model=self.model,
                    attempts=attempt,
                    data=validated,
                    raw_json=validated.model_dump_json(indent=2),
                )
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                repair_turns.append((cleaned_json, str(exc)))

        raise AnalysisError(
            f"Failed strict {schema_cls.__name__} validation after "
            f"{self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error
