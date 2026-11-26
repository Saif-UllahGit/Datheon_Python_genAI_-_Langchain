#!/usr/bin/env python3
"""Executable CLI for parsing unstructured invoices and resumes into strict JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure local src/ package is importable when executed directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from doc_analyzer.analyzer import AnalysisError, DocumentAnalyzer
from doc_analyzer.prompts import PromptParameters, build_prompts
from doc_analyzer.schemas import DocumentType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse unstructured invoice or resume text into strictly validated JSON."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=[t.value for t in DocumentType],
        help="Document type to parse: 'invoice' or 'resume'.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        type=str,
        help="Path to a UTF-8 text file, or '-' to read from standard input.",
    )
    source_group.add_argument(
        "--text",
        type=str,
        help="Inline unstructured document text.",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "mock"],
        help="LLM backend to use (default: gemini).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name (defaults to provider's recommended model).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override (defaults to GEMINI_API_KEY / OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic extraction).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max repair retries if output fails Pydantic validation (default: 2).",
    )
    parser.add_argument(
        "--currency",
        default="USD",
        help="Fallback ISO-4217 currency for invoices when unspecified (default: USD).",
    )
    parser.add_argument(
        "--target-role",
        default=None,
        help="Optional target role to evaluate candidate fit during resume analysis.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language for descriptive output fields (default: English).",
    )
    parser.add_argument(
        "--extra-instructions",
        default="",
        help="Additional parameterized prompt instructions.",
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples in the prompt.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered parameterized prompt and exit without calling an LLM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write the validated JSON output.",
    )
    return parser.parse_args(argv)


def load_document_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input == "-":
        return sys.stdin.read()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return input_path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        document_text = load_document_text(args)
        params = PromptParameters(
            document_type=DocumentType(args.type),
            default_currency=args.currency,
            target_role=args.target_role,
            output_language=args.language,
            include_few_shot=not args.no_few_shot,
            extra_instructions=args.extra_instructions,
        )

        if args.show_prompt:
            rendered = build_prompts(document_text, params)
            print("=== SYSTEM INSTRUCTION ===")
            print(rendered.system_instruction)
            print("\n=== USER PROMPT (WITH FEW-SHOTS) ===")
            print(rendered.as_single_user_prompt())
            return 0

        analyzer = DocumentAnalyzer(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        result = analyzer.analyze(document_text, params)
    except (FileNotFoundError, ValueError, AnalysisError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.raw_json + "\n", encoding="utf-8")
        print(
            f"Validated {result.document_type.value} JSON written to {args.output} "
            f"(provider={result.provider}, model={result.model}, attempts={result.attempts})",
            file=sys.stderr,
        )
    else:
        print(result.raw_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
