"""Core entity types for microdata representation.

Entities represent the hierarchical structure of tax-benefit microdata:
- Person: Individual-level attributes
- TaxUnit: Tax filing unit (IRS perspective)
- Household: Residential unit (housing costs, geography)
- Family: Family grouping used by some systems
- BenefitUnit: Benefit assessment unit (UK-style family benefit unit)
- SPMUnit: Supplemental Poverty Measure unit
- Record: Sub-person records (W-2s, K-1s, 1099s, etc.)
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(Enum):
    """Types of entities in the microdata hierarchy."""

    RECORD = "record"
    PERSON = "person"
    TAX_UNIT = "tax_unit"
    HOUSEHOLD = "household"
    FAMILY = "family"
    BENEFIT_UNIT = "benefit_unit"
    SPM_UNIT = "spm_unit"

    @property
    def level(self) -> int:
        """Hierarchy level (0 = lowest/most granular)."""
        levels = {
            EntityType.RECORD: 0,
            EntityType.PERSON: 1,
            EntityType.TAX_UNIT: 2,
            EntityType.HOUSEHOLD: 2,
            EntityType.FAMILY: 2,
            EntityType.BENEFIT_UNIT: 2,
            EntityType.SPM_UNIT: 2,
        }
        return levels[self]

    @property
    def is_group(self) -> bool:
        """Whether this entity groups persons."""
        return self not in (EntityType.RECORD, EntityType.PERSON)


class FilingStatus(Enum):
    """Tax filing status options."""

    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"
    MARRIED_FILING_SEPARATELY = "married_filing_separately"
    HEAD_OF_HOUSEHOLD = "head_of_household"
    QUALIFYING_WIDOW = "qualifying_widow"


class RecordType(Enum):
    """Types of sub-person records (tax forms, etc.)."""

    W2 = "w2"
    K1 = "k1"
    FORM_1099_INT = "1099_int"
    FORM_1099_DIV = "1099_div"
    FORM_1099_MISC = "1099_misc"
    FORM_1099_NEC = "1099_nec"
    FORM_1099_R = "1099_r"
    FORM_1099_G = "1099_g"
    FORM_1099_SSA = "1099_ssa"
    SCHEDULE_C = "schedule_c"
    SCHEDULE_E = "schedule_e"
    SCHEDULE_F = "schedule_f"


class Entity(BaseModel):
    """Base class for all entities."""

    id: str = Field(..., description="Unique identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (source, imputation flags, etc.)",
    )

    @property
    def entity_type(self) -> EntityType:
        """Override in subclasses."""
        raise NotImplementedError

    model_config = {"frozen": False, "extra": "allow"}


class Person(Entity):
    """Individual person in the microdata.

    Persons are the atomic unit. They belong to group entities
    (tax units, households, etc.) and can have multiple records
    (W-2s, 1099s, etc.).
    """

    # Demographics
    age: int | None = Field(default=None, ge=0, le=130)
    is_male: bool | None = None
    is_citizen: bool | None = None
    is_disabled: bool | None = None
    is_blind: bool | None = None
    marital_status: str | None = None
    education_level: str | None = None

    # Income (annual, may have intrayear distribution)
    employment_income: float | None = None
    self_employment_income: float | None = None
    pension_income: float | None = None
    social_security_income: float | None = None
    interest_income: float | None = None
    dividend_income: float | None = None
    capital_gains: float | None = None
    rental_income: float | None = None
    unemployment_income: float | None = None

    # Group memberships
    tax_unit_id: str | None = None
    household_id: str | None = None
    family_id: str | None = None
    spm_unit_id: str | None = None

    # Sub-person records
    record_ids: list[str] = Field(default_factory=list)

    @property
    def entity_type(self) -> EntityType:
        return EntityType.PERSON


class TaxUnit(Entity):
    """Tax filing unit (corresponds to a tax return).

    Contains one or more persons who file together.
    """

    # Members
    filer_id: str | None = None
    spouse_id: str | None = None
    member_ids: list[str] = Field(default_factory=list)

    # Filing attributes
    filing_status: FilingStatus | None = None

    # Aggregated income (from members)
    adjusted_gross_income: float | None = None
    taxable_income: float | None = None
    federal_income_tax: float | None = None

    # Deductions
    itemized_deductions: float | None = None
    charitable_deductions: float | None = None
    mortgage_interest_deduction: float | None = None
    salt_deduction: float | None = None

    # Credits
    earned_income_credit: float | None = None
    child_tax_credit: float | None = None
    child_care_credit: float | None = None

    @property
    def entity_type(self) -> EntityType:
        return EntityType.TAX_UNIT


class Household(Entity):
    """Census household (housing unit).

    Contains one or more persons living together.
    Primary entity for housing costs and geographic location.
    """

    # Members
    householder_id: str | None = None
    member_ids: list[str] = Field(default_factory=list)

    # Geography
    state_fips: str | None = Field(default=None, pattern=r"^\d{2}$")
    county_fips: str | None = Field(default=None, pattern=r"^\d{3}$")
    zcta: str | None = Field(default=None, pattern=r"^\d{5}$")
    puma: str | None = None
    congressional_district: str | None = None

    # Housing
    is_owner: bool | None = None
    rent: float | None = None
    mortgage_payment: float | None = None
    property_tax: float | None = None
    home_value: float | None = None

    # Weight
    weight: float = Field(default=1.0, ge=0.0)

    @property
    def entity_type(self) -> EntityType:
        return EntityType.HOUSEHOLD


class Family(Entity):
    """SPM family unit for poverty calculations."""

    member_ids: list[str] = Field(default_factory=list)
    head_id: str | None = None

    @property
    def entity_type(self) -> EntityType:
        return EntityType.FAMILY


class BenefitUnit(Entity):
    """Benefit assessment unit.

    Used by tax-benefit systems that determine eligibility/resources at a
    family-benefit-unit grain distinct from households or tax units.
    """

    member_ids: list[str] = Field(default_factory=list)
    head_id: str | None = None

    @property
    def entity_type(self) -> EntityType:
        return EntityType.BENEFIT_UNIT


class SPMUnit(Entity):
    """Supplemental Poverty Measure unit."""

    member_ids: list[str] = Field(default_factory=list)
    head_id: str | None = None

    # SPM resources and thresholds
    spm_resources: float | None = None
    spm_threshold: float | None = None

    @property
    def entity_type(self) -> EntityType:
        return EntityType.SPM_UNIT


class Record(Entity):
    """Sub-person record (tax form, information return, etc.).

    A person can have multiple records of the same type
    (e.g., multiple W-2s from different employers).
    """

    person_id: str = Field(..., description="Person this record belongs to")
    record_type: RecordType = Field(..., description="Type of record")

    # Common across record types
    tax_year: int | None = None
    employer_id: str | None = None  # EIN for W-2
    payer_id: str | None = None  # For 1099s

    # W-2 fields
    wages: float | None = None
    federal_tax_withheld: float | None = None
    social_security_wages: float | None = None
    medicare_wages: float | None = None
    state_wages: float | None = None
    state_tax_withheld: float | None = None

    # K-1 fields (Schedule K-1 from partnerships/S-corps)
    ordinary_business_income: float | None = None
    guaranteed_payments: float | None = None
    interest_income_k1: float | None = None
    dividends_k1: float | None = None
    capital_gains_k1: float | None = None
    rental_income_k1: float | None = None
    royalties_k1: float | None = None

    # 1099-INT fields
    interest_income_1099: float | None = None
    tax_exempt_interest: float | None = None

    # 1099-DIV fields
    ordinary_dividends: float | None = None
    qualified_dividends: float | None = None
    capital_gain_distributions: float | None = None

    # 1099-R fields (retirement distributions)
    gross_distribution: float | None = None
    taxable_amount: float | None = None
    federal_tax_withheld_1099r: float | None = None

    # Schedule C fields (self-employment)
    gross_receipts: float | None = None
    cost_of_goods_sold: float | None = None
    net_profit: float | None = None

    @property
    def entity_type(self) -> EntityType:
        return EntityType.RECORD
