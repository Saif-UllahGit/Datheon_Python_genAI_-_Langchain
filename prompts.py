"""Parameterized prompt templates with few-shot examples for document extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from string import Template
from typing import Any

from .schemas import DocumentType, get_schema_for_type


@dataclass(frozen=True)
class FewShotExample:
    """Pair of noisy unstructured input text and its strictly valid target JSON."""

    label: str
    raw_text: str
    expected_json: dict[str, Any]


@dataclass(frozen=True)
class PromptParameters:
    """Parameters used to customize prompt generation per request."""

    document_type: DocumentType
    default_currency: str = "USD"
    target_role: str | None = None
    output_language: str = "English"
    strict_nulls: bool = True
    include_few_shot: bool = True
    max_few_shot_examples: int = 2
    extra_instructions: str = ""
    custom_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedPrompt:
    """Rendered system instruction, few-shot turns, and user prompt."""

    system_instruction: str
    few_shot_examples: tuple[FewShotExample, ...]
    user_prompt: str

    def as_openai_messages(self) -> list[dict[str, str]]:
        """Format as OpenAI chat messages with few-shot assistant turns."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_instruction}
        ]
        for example in self.few_shot_examples:
            messages.append(
                {
                    "role": "user",
                    "content": f"Extract structured JSON from this document:\n\n{example.raw_text}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example.expected_json, indent=2),
                }
            )
        messages.append({"role": "user", "content": self.user_prompt})
        return messages

    def as_single_user_prompt(self) -> str:
        """Render few-shot examples inline for providers without turn-based priming."""
        sections: list[str] = []
        if self.few_shot_examples:
            sections.append("### FEW-SHOT EXAMPLES")
            for idx, example in enumerate(self.few_shot_examples, start=1):
                sections.append(
                    f"Example {idx} ({example.label}):\n"
                    f"INPUT:\n{example.raw_text}\n\n"
                    f"OUTPUT JSON:\n{json.dumps(example.expected_json, indent=2)}"
                )
        sections.append("### TARGET TASK\n" + self.user_prompt)
        return "\n\n".join(sections)


_SYSTEM_TEMPLATE = Template(
    """You are an expert structured data extraction engine specialized in ${doc_type_label} analysis.
Your sole task is to parse noisy, unstructured text into a single JSON object that strictly conforms to the provided JSON Schema.

RULES:
1. Output ONLY valid JSON — no markdown fences, no commentary, no trailing text.
2. Conform strictly to every field name, data type, enum, and constraint in the JSON Schema. Do not invent extra keys.
3. Write descriptive text fields in ${output_language}.
4. ${null_handling_rule}
5. ${domain_specific_rules}
${extra_rules_block}
JSON SCHEMA:
${json_schema}"""
)

_USER_TEMPLATE = Template(
    """Analyze the following unstructured ${doc_type_name} text and return the strictly validated JSON object.
${context_block}
DOCUMENT TEXT:
\"\"\"
${document_text}
\"\"\""""
)


_INVOICE_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        label="Noisy service invoice with discount and tax",
        raw_text=(
            "NORTHSTAR CLOUD OPS LLC | VAT: US-94-3318201 | billing@northstar.io\n"
            "742 Market St, San Francisco, CA\n"
            "INVOICE #NS-2026-019  |  Issued: 14 Feb 2026  |  Terms: Net 15 (Due 2026-03-01)\n"
            "Bill To: Acme Robotics Inc., 88 Pine Rd, Austin TX (ap@acmerobotics.com)\n"
            "Items:\n"
            "- Kubernetes Cluster Mgmt (20 hrs @ $125.00) -> $2,500.00 [tax 8%]\n"
            "- Dedicated GPU Node Setup x 1 -> $600.00 [tax 8%]\n"
            "Subtotal: $3,100.00 | Loyalty Discount: -$100.00 | Tax (8% on 3100): $248.00\n"
            "TOTAL DUE: $3,248.00  [STATUS: UNPAID]"
        ),
        expected_json={
            "invoice_number": "NS-2026-019",
            "issue_date": "2026-02-14",
            "due_date": "2026-03-01",
            "currency": "USD",
            "vendor": {
                "name": "Northstar Cloud Ops LLC",
                "tax_id": "US-94-3318201",
                "email": "billing@northstar.io",
                "phone": None,
                "address": "742 Market St, San Francisco, CA",
            },
            "customer": {
                "name": "Acme Robotics Inc.",
                "tax_id": None,
                "email": "ap@acmerobotics.com",
                "phone": None,
                "address": "88 Pine Rd, Austin TX",
            },
            "line_items": [
                {
                    "description": "Kubernetes Cluster Mgmt",
                    "quantity": 20.0,
                    "unit_price": 125.0,
                    "tax_rate_percent": 8.0,
                    "line_total": 2500.0,
                },
                {
                    "description": "Dedicated GPU Node Setup",
                    "quantity": 1.0,
                    "unit_price": 600.0,
                    "tax_rate_percent": 8.0,
                    "line_total": 600.0,
                },
            ],
            "subtotal": 3100.0,
            "tax_amount": 248.0,
            "discount_amount": 100.0,
            "total_amount": 3248.0,
            "payment_terms": "Net 15",
            "payment_status": "unpaid",
            "confidence_notes": [],
        },
    ),
    FewShotExample(
        label="Informal EUR receipt with missing due date",
        raw_text=(
            "Studio K Design - Berlin | hello@studiok.de | +49 30 5551234\n"
            "Ref: INV_884 | Date: 03/05/2026 | PAID via wire\n"
            "Client: FinPulse GmbH\n"
            "1x Brand Guidelines Refresh = EUR 1800.00 (0% VAT reverse charge)\n"
            "Total Paid: 1800.00 EUR"
        ),
        expected_json={
            "invoice_number": "INV_884",
            "issue_date": "2026-05-03",
            "due_date": None,
            "currency": "EUR",
            "vendor": {
                "name": "Studio K Design",
                "tax_id": None,
                "email": "hello@studiok.de",
                "phone": "+49 30 5551234",
                "address": "Berlin",
            },
            "customer": {
                "name": "FinPulse GmbH",
                "tax_id": None,
                "email": None,
                "phone": None,
                "address": None,
            },
            "line_items": [
                {
                    "description": "Brand Guidelines Refresh",
                    "quantity": 1.0,
                    "unit_price": 1800.0,
                    "tax_rate_percent": 0.0,
                    "line_total": 1800.0,
                }
            ],
            "subtotal": 1800.0,
            "tax_amount": 0.0,
            "discount_amount": 0.0,
            "total_amount": 1800.0,
            "payment_terms": None,
            "payment_status": "paid",
            "confidence_notes": [
                "Interpreted 03/05/2026 as DD/MM/YYYY given German vendor location.",
                "Due date not stated on paid invoice.",
            ],
        },
    ),
)


_RESUME_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        label="Unstructured engineering resume with target role fit",
        raw_text=(
            "PRIYA NAIR\n"
            "priya.nair@example.com | Seattle, WA | github.com/priyanair\n"
            "About: Backend engineer with 5 years building high-throughput Python & Go services.\n"
            "Work:\n"
            "* Senior Software Engineer @ NimbusPay (Mar 2023 - Present, Remote)\n"
            "  - Rebuilt webhook dispatcher in Python/FastAPI reducing p99 latency by 42%.\n"
            "  - Managed PostgreSQL and Redis clusters on AWS.\n"
            "* Software Engineer @ DataSprout (Jun 2021 - Feb 2023, Seattle)\n"
            "  - Built ETL pipelines with Airflow and Docker.\n"
            "Education: B.S. Computer Science, University of Washington, 2021 (GPA 3.8)\n"
            "Skills: Python, Go, SQL, FastAPI, Docker, PostgreSQL, Redis, AWS\n"
            "Certs: AWS Certified Solutions Architect - Associate"
        ),
        expected_json={
            "candidate_name": "Priya Nair",
            "contact": {
                "email": "priya.nair@example.com",
                "phone": None,
                "location": "Seattle, WA",
                "linkedin_url": None,
                "github_url": "https://github.com/priyanair",
                "portfolio_url": None,
            },
            "professional_summary": (
                "Backend engineer with 5 years of experience building high-throughput "
                "Python and Go services, distributed data pipelines, and cloud infrastructure."
            ),
            "years_of_experience": 5.0,
            "overall_seniority": "senior",
            "skills": {
                "programming_languages": ["Python", "Go", "SQL"],
                "frameworks_and_tools": ["FastAPI", "Docker", "Airflow", "PostgreSQL", "Redis", "AWS"],
                "domain_skills": ["Backend Engineering", "ETL Pipelines", "Latency Optimization"],
                "spoken_languages": [],
            },
            "work_experience": [
                {
                    "company": "NimbusPay",
                    "role": "Senior Software Engineer",
                    "start_date": "2023-03",
                    "end_date": None,
                    "is_current": True,
                    "location": "Remote",
                    "highlights": [
                        "Rebuilt webhook dispatcher in Python/FastAPI reducing p99 latency by 42%.",
                        "Managed PostgreSQL and Redis clusters on AWS.",
                    ],
                },
                {
                    "company": "DataSprout",
                    "role": "Software Engineer",
                    "start_date": "2021-06",
                    "end_date": "2023-02",
                    "is_current": False,
                    "location": "Seattle",
                    "highlights": ["Built ETL pipelines with Airflow and Docker."],
                },
            ],
            "education": [
                {
                    "institution": "University of Washington",
                    "degree": "B.S.",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2021,
                    "gpa": "3.8",
                }
            ],
            "certifications": ["AWS Certified Solutions Architect - Associate"],
            "projects": [],
            "role_fit_assessment": None,
            "confidence_notes": ["Normalized github.com/priyanair to full https URL."],
        },
    ),
)


_FEW_SHOT_REGISTRY: dict[DocumentType, tuple[FewShotExample, ...]] = {
    DocumentType.INVOICE: _INVOICE_EXAMPLES,
    DocumentType.RESUME: _RESUME_EXAMPLES,
}


def build_prompts(document_text: str, params: PromptParameters) -> RenderedPrompt:
    """Build parameterized system and user prompts with optional few-shot examples."""
    if not document_text or not document_text.strip():
        raise ValueError("document_text must be a non-empty string.")

    schema_cls = get_schema_for_type(params.document_type)
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)

    null_rule = (
        "Set missing optional fields explicitly to null rather than guessing values."
        if params.strict_nulls
        else "Infer missing fields only when strongly implied and log inferences in confidence_notes."
    )

    if params.document_type == DocumentType.INVOICE:
        doc_type_label = "financial invoice"
        domain_rules = (
            f"Normalize dates to YYYY-MM-DD, currency to 3-letter ISO 4217 "
            f"(default to {params.default_currency.upper()} only if no currency symbol/code appears), "
            f"and ensure total_amount == subtotal + tax_amount - discount_amount."
        )
    else:
        doc_type_label = "candidate resume / CV"
        if params.target_role:
            domain_rules = (
                f"Normalize employment dates to YYYY or YYYY-MM, set end_date to null when "
                f"is_current is true, and populate role_fit_assessment against target_role="
                f"'{params.target_role}'."
            )
        else:
            domain_rules = (
                "Normalize employment dates to YYYY or YYYY-MM, set end_date to null when "
                "is_current is true, and set role_fit_assessment to null."
            )

    extra_rules_block = (
        f"6. Additional instructions: {params.extra_instructions.strip()}\n"
        if params.extra_instructions.strip()
        else ""
    )

    system_instruction = _SYSTEM_TEMPLATE.substitute(
        doc_type_label=doc_type_label,
        output_language=params.output_language,
        null_handling_rule=null_rule,
        domain_specific_rules=domain_rules,
        extra_rules_block=extra_rules_block,
        json_schema=schema_json,
    )

    context_lines: list[str] = []
    if params.document_type == DocumentType.INVOICE:
        context_lines.append(f"- Default fallback currency: {params.default_currency.upper()}")
    if params.document_type == DocumentType.RESUME and params.target_role:
        context_lines.append(f"- Target role for fit evaluation: {params.target_role}")
    for key, val in params.custom_metadata.items():
        context_lines.append(f"- {key}: {val}")

    context_block = (
        "EXTRACTION PARAMETERS:\n" + "\n".join(context_lines) + "\n"
        if context_lines
        else ""
    )

    user_prompt = _USER_TEMPLATE.substitute(
        doc_type_name=params.document_type.value,
        context_block=context_block,
        document_text=document_text.strip(),
    )

    examples: tuple[FewShotExample, ...] = ()
    if params.include_few_shot and params.max_few_shot_examples > 0:
        examples = _FEW_SHOT_REGISTRY[params.document_type][: params.max_few_shot_examples]

    return RenderedPrompt(
        system_instruction=system_instruction,
        few_shot_examples=examples,
        user_prompt=user_prompt,
    )
