"""Structured document analysis package using Gemini/OpenAI and Pydantic."""

from .analyzer import AnalysisError, DocumentAnalyzer
from .prompts import PromptParameters, build_prompts
from .schemas import (
    DocumentType,
    InvoiceAnalysis,
    ResumeAnalysis,
    get_schema_for_type,
)

__all__ = [
    "AnalysisError",
    "DocumentAnalyzer",
    "DocumentType",
    "InvoiceAnalysis",
    "PromptParameters",
    "ResumeAnalysis",
    "build_prompts",
    "get_schema_for_type",
]
