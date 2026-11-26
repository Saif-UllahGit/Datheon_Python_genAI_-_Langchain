"""Strict Pydantic v2 schemas for invoice and resume analysis."""

from __future__ import annotations

from datetime import date
from enum import Enum
import re
from typing import Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(str, Enum):
    """Supported unstructured document categories."""

    INVOICE = "invoice"
    RESUME = "resume"


class StrictModel(BaseModel):
    """Base model forbidding unexpected fields and trimming string whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Invoice Analysis Schemas
# ---------------------------------------------------------------------------


class PartyInfo(StrictModel):
    """Vendor or customer identity details extracted from an invoice."""

    name: str = Field(..., min_length=1, description="Legal or trading name of the party.")
    tax_id: Optional[str] = Field(
        default=None, description="VAT, EIN, GST, or tax registration identifier."
    )
    email: Optional[str] = Field(default=None, description="Contact email address.")
    phone: Optional[str] = Field(default=None, description="Contact phone number.")
    address: Optional[str] = Field(default=None, description="Mailing or billing address.")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and "@" not in value:
            raise ValueError(f"Invalid email format: {value!r}")
        return value


class LineItem(StrictModel):
    """Single billed line item on an invoice."""

    description: str = Field(..., min_length=1, description="Item or service description.")
    quantity: float = Field(..., gt=0, description="Units billed (must be positive).")
    unit_price: float = Field(..., ge=0, description="Price per unit before tax.")
    tax_rate_percent: float = Field(
        default=0.0, ge=0, le=100, description="Applicable tax rate percentage (0-100)."
    )
    line_total: float = Field(..., ge=0, description="Total amount for this line item.")


class InvoiceAnalysis(StrictModel):
    """Strictly validated structured extraction of an unstructured invoice."""

    invoice_number: str = Field(..., min_length=1, description="Unique invoice identifier.")
    issue_date: date = Field(..., description="Invoice issue date in YYYY-MM-DD format.")
    due_date: Optional[date] = Field(
        default=None, description="Payment due date in YYYY-MM-DD format."
    )
    currency: str = Field(
        ..., min_length=3, max_length=3, description="3-letter ISO 4217 currency code."
    )
    vendor: PartyInfo = Field(..., description="Issuing vendor or supplier.")
    customer: PartyInfo = Field(..., description="Billed customer or recipient.")
    line_items: list[LineItem] = Field(
        ..., min_length=1, description="List of billed products or services."
    )
    subtotal: float = Field(..., ge=0, description="Sum of line items before tax and discount.")
    tax_amount: float = Field(default=0.0, ge=0, description="Total tax charged.")
    discount_amount: float = Field(default=0.0, ge=0, description="Total discount applied.")
    total_amount: float = Field(..., ge=0, description="Final payable invoice amount.")
    payment_terms: Optional[str] = Field(
        default=None, description="Payment terms such as 'Net 30' or 'Due on receipt'."
    )
    payment_status: Literal["paid", "unpaid", "partial", "overdue", "unknown"] = Field(
        default="unknown", description="Normalized payment status."
    )
    confidence_notes: list[str] = Field(
        default_factory=list,
        description="Notes on ambiguous, inferred, or missing fields in the source text.",
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_code(cls, value: str) -> str:
        code = value.upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError(f"Currency must be a 3-letter ISO code, got {value!r}")
        return code

    @model_validator(mode="after")
    def validate_dates_and_totals(self) -> "InvoiceAnalysis":
        if self.due_date is not None and self.due_date < self.issue_date:
            raise ValueError(
                f"due_date ({self.due_date}) cannot be earlier than issue_date ({self.issue_date})"
            )

        expected_total = round(self.subtotal + self.tax_amount - self.discount_amount, 2)
        if abs(expected_total - self.total_amount) > 0.05:
            raise ValueError(
                f"Invoice total_amount ({self.total_amount:.2f}) does not match "
                f"subtotal + tax_amount - discount_amount ({expected_total:.2f})"
            )
        return self


# ---------------------------------------------------------------------------
# Resume Analysis Schemas
# ---------------------------------------------------------------------------


class ContactInfo(StrictModel):
    """Candidate contact and profile links."""

    email: Optional[str] = Field(default=None, description="Candidate email address.")
    phone: Optional[str] = Field(default=None, description="Candidate phone number.")
    location: Optional[str] = Field(default=None, description="City, state, or country.")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL.")
    github_url: Optional[str] = Field(default=None, description="GitHub profile URL.")
    portfolio_url: Optional[str] = Field(default=None, description="Personal website URL.")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and "@" not in value:
            raise ValueError(f"Invalid email format: {value!r}")
        return value


class SkillBreakdown(StrictModel):
    """Categorized candidate skills."""

    programming_languages: list[str] = Field(default_factory=list)
    frameworks_and_tools: list[str] = Field(default_factory=list)
    domain_skills: list[str] = Field(default_factory=list)
    spoken_languages: list[str] = Field(default_factory=list)


class WorkExperience(StrictModel):
    """Single employment entry."""

    company: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    start_date: str = Field(
        ..., min_length=4, description="Start date in YYYY-MM or YYYY format."
    )
    end_date: Optional[str] = Field(
        default=None, description="End date in YYYY-MM or YYYY format, or null if current."
    )
    is_current: bool = Field(default=False)
    location: Optional[str] = Field(default=None)
    highlights: list[str] = Field(
        default_factory=list, description="Key measurable achievements and responsibilities."
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_year_month(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not re.fullmatch(r"\d{4}(-\d{2})?", value):
            raise ValueError(f"Date must use YYYY or YYYY-MM format, got {value!r}")
        return value

    @model_validator(mode="after")
    def validate_current_flag(self) -> "WorkExperience":
        if self.is_current and self.end_date is not None:
            raise ValueError("end_date must be null when is_current is true")
        return self


class Education(StrictModel):
    """Academic degree or training record."""

    institution: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)
    field_of_study: Optional[str] = Field(default=None)
    graduation_year: Optional[int] = Field(default=None, ge=1950, le=2035)
    gpa: Optional[str] = Field(default=None)


class Project(StrictModel):
    """Notable technical or professional project."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    technologies: list[str] = Field(default_factory=list)


class RoleFitAssessment(StrictModel):
    """Optional evaluation of candidate fit against a parameterized target role."""

    target_role: str = Field(..., min_length=1)
    match_score: int = Field(..., ge=0, le=100, description="Estimated fit score (0-100).")
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ResumeAnalysis(StrictModel):
    """Strictly validated structured extraction of an unstructured resume/CV."""

    candidate_name: str = Field(..., min_length=1)
    contact: ContactInfo
    professional_summary: str = Field(..., min_length=1)
    years_of_experience: float = Field(..., ge=0, le=70)
    overall_seniority: Literal[
        "intern", "junior", "mid", "senior", "staff", "lead", "executive"
    ]
    skills: SkillBreakdown
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    role_fit_assessment: Optional[RoleFitAssessment] = Field(
        default=None,
        description="Populated only when a target_role parameter is provided.",
    )
    confidence_notes: list[str] = Field(
        default_factory=list,
        description="Notes on ambiguous, inferred, or missing fields in the source text.",
    )


SCHEMA_REGISTRY: dict[DocumentType, Type[StrictModel]] = {
    DocumentType.INVOICE: InvoiceAnalysis,
    DocumentType.RESUME: ResumeAnalysis,
}


def get_schema_for_type(doc_type: DocumentType | str) -> Type[StrictModel]:
    """Return the Pydantic model class corresponding to the given document type."""
    normalized = DocumentType(doc_type)
    return SCHEMA_REGISTRY[normalized]
