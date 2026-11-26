"""Unit and integration tests for schemas, parameterized prompts, and validation retries."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import analyze  # noqa: E402
from doc_analyzer.analyzer import (  # noqa: E402
    AnalysisError,
    BaseLLMClient,
    DocumentAnalyzer,
    strip_markdown_json_fences,
)
from doc_analyzer.prompts import (  # noqa: E402
    _INVOICE_EXAMPLES,
    _RESUME_EXAMPLES,
    PromptParameters,
    RenderedPrompt,
    build_prompts,
)
from doc_analyzer.schemas import (  # noqa: E402
    DocumentType,
    InvoiceAnalysis,
    ResumeAnalysis,
)


class ScriptedLLMClient(BaseLLMClient):
    """Test double returning scripted responses and recording repair turns."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[tuple[str, str]] | None] = []

    def generate_json(
        self,
        rendered_prompt: RenderedPrompt,
        schema_cls,
        repair_turns: list[tuple[str, str]] | None = None,
    ) -> str:
        self.calls.append(list(repair_turns) if repair_turns else None)
        if not self._responses:
            raise AssertionError("No scripted responses left")
        return self._responses.pop(0)


class SchemaValidationTests(unittest.TestCase):
    def test_few_shot_examples_satisfy_strict_schemas(self) -> None:
        for example in _INVOICE_EXAMPLES:
            parsed = InvoiceAnalysis.model_validate(example.expected_json)
            self.assertEqual(len(parsed.currency), 3)

        for example in _RESUME_EXAMPLES:
            parsed = ResumeAnalysis.model_validate(example.expected_json)
            self.assertGreaterEqual(parsed.years_of_experience, 0)

    def test_invoice_rejects_mismatched_total_and_extra_fields(self) -> None:
        bad_payload = dict(_INVOICE_EXAMPLES[0].expected_json)
        bad_payload["total_amount"] = 9999.99
        with self.assertRaises(Exception):
            InvoiceAnalysis.model_validate(bad_payload)

        extra_payload = dict(_INVOICE_EXAMPLES[0].expected_json)
        extra_payload["unexpected_key"] = "boom"
        with self.assertRaises(Exception):
            InvoiceAnalysis.model_validate(extra_payload)

    def test_invoice_rejects_due_date_before_issue_date(self) -> None:
        bad_dates = dict(_INVOICE_EXAMPLES[0].expected_json)
        bad_dates["issue_date"] = "2026-05-10"
        bad_dates["due_date"] = "2026-05-01"
        with self.assertRaises(Exception):
            InvoiceAnalysis.model_validate(bad_dates)

    def test_resume_rejects_end_date_when_is_current(self) -> None:
        bad_resume = json.loads(json.dumps(_RESUME_EXAMPLES[0].expected_json))
        bad_resume["work_experience"][0]["is_current"] = True
        bad_resume["work_experience"][0]["end_date"] = "2026-01"
        with self.assertRaises(Exception):
            ResumeAnalysis.model_validate(bad_resume)


class PromptTemplateTests(unittest.TestCase):
    def test_parameterized_invoice_prompt_rendering(self) -> None:
        params = PromptParameters(
            document_type=DocumentType.INVOICE,
            default_currency="eur",
            output_language="English",
            extra_instructions="Always preserve VAT IDs verbatim.",
        )
        rendered = build_prompts("Invoice #99 for consulting", params)
        self.assertIn("EUR", rendered.system_instruction)
        self.assertIn("Always preserve VAT IDs verbatim.", rendered.system_instruction)
        self.assertEqual(len(rendered.few_shot_examples), 2)
        messages = rendered.as_openai_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")

    def test_parameterized_resume_prompt_with_target_role(self) -> None:
        params = PromptParameters(
            document_type=DocumentType.RESUME,
            target_role="Staff Backend Engineer",
            include_few_shot=False,
        )
        rendered = build_prompts("Jane Doe - Python Engineer", params)
        self.assertIn("Staff Backend Engineer", rendered.system_instruction)
        self.assertIn("Staff Backend Engineer", rendered.user_prompt)
        self.assertEqual(len(rendered.few_shot_examples), 0)


class AnalyzerRetryTests(unittest.TestCase):
    def test_strip_markdown_json_fences(self) -> None:
        wrapped = 'Here is the JSON:\n```json\n{"a": 1}\n```\n'
        self.assertEqual(strip_markdown_json_fences(wrapped), '{"a": 1}')

    def test_repair_retry_recovers_from_invalid_first_response(self) -> None:
        valid_json = json.dumps(_INVOICE_EXAMPLES[0].expected_json)
        invalid_payload = dict(_INVOICE_EXAMPLES[0].expected_json)
        invalid_payload["currency"] = "DOLLARS"  # Violates 3-letter ISO constraint
        fenced_valid = f"```json\n{valid_json}\n```"

        client = ScriptedLLMClient([json.dumps(invalid_payload), fenced_valid])
        analyzer = DocumentAnalyzer(provider="gemini", max_retries=2, client=client)
        result = analyzer.analyze(
            "Sample noisy invoice text",
            PromptParameters(document_type=DocumentType.INVOICE),
        )

        self.assertEqual(result.attempts, 2)
        self.assertIsInstance(result.data, InvoiceAnalysis)
        self.assertEqual(result.data.currency, "USD")
        self.assertIsNone(client.calls[0])
        self.assertEqual(len(client.calls[1]), 1)

    def test_raises_analysis_error_when_retries_exhausted(self) -> None:
        client = ScriptedLLMClient(["not json", "{still bad}"])
        analyzer = DocumentAnalyzer(provider="openai", max_retries=1, client=client)
        with self.assertRaises(AnalysisError):
            analyzer.analyze(
                "Sample resume text",
                PromptParameters(document_type=DocumentType.RESUME),
            )


class CLITests(unittest.TestCase):
    def test_cli_mock_invoice_and_resume_end_to_end(self) -> None:
        invoice_sample = PROJECT_ROOT / "samples" / "messy_invoice.txt"
        resume_sample = PROJECT_ROOT / "samples" / "messy_resume.txt"

        out_dir = PROJECT_ROOT / ".test_tmp"
        out_dir.mkdir(exist_ok=True)
        inv_out = out_dir / "invoice.json"
        res_out = out_dir / "resume.json"

        try:
            rc1 = analyze.main(
                [
                    "--type",
                    "invoice",
                    "--input",
                    str(invoice_sample),
                    "--provider",
                    "mock",
                    "--output",
                    str(inv_out),
                ]
            )
            self.assertEqual(rc1, 0)
            InvoiceAnalysis.model_validate_json(inv_out.read_text(encoding="utf-8"))

            rc2 = analyze.main(
                [
                    "--type",
                    "resume",
                    "--input",
                    str(resume_sample),
                    "--provider",
                    "mock",
                    "--target-role",
                    "Staff ML Platform Engineer",
                    "--output",
                    str(res_out),
                ]
            )
            self.assertEqual(rc2, 0)
            parsed_res = ResumeAnalysis.model_validate_json(res_out.read_text(encoding="utf-8"))
            self.assertIsNotNone(parsed_res.role_fit_assessment)
            self.assertEqual(
                parsed_res.role_fit_assessment.target_role,
                "Staff ML Platform Engineer",
            )
        finally:
            for p in (inv_out, res_out):
                if p.exists():
                    p.unlink()
            if out_dir.exists():
                out_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
