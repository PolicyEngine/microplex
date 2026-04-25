"""Time period handling for microdata.

Supports multiple granularities (day, month, quarter, year) with
arithmetic operations, containment checks, and string parsing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from enum import Enum
from functools import total_ordering

from pydantic import BaseModel, Field, model_validator


class PeriodType(Enum):
    """Granularity of time periods."""

    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

    @property
    def days(self) -> int:
        """Approximate days in this period type."""
        return {
            PeriodType.DAY: 1,
            PeriodType.MONTH: 30,
            PeriodType.QUARTER: 91,
            PeriodType.YEAR: 365,
        }[self]

    def contains(self, other: PeriodType) -> bool:
        """Check if this period type contains another."""
        return self.days > other.days


@total_ordering
class Period(BaseModel):
    """A specific time period at a given granularity.

    Examples:
        >>> Period.of_year(2024)
        Period(year=2024, period_type=YEAR)
        >>> Period.of_month(2024, 3)
        Period(year=2024, month=3, period_type=MONTH)
        >>> Period.of_quarter(2024, 2)
        Period(year=2024, quarter=2, period_type=QUARTER)
    """

    year: int = Field(..., ge=1900, le=2200)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)
    quarter: int | None = Field(default=None, ge=1, le=4)
    period_type: PeriodType = PeriodType.YEAR

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_period_consistency(self) -> Period:
        """Ensure period components match period type."""
        if self.period_type == PeriodType.YEAR:
            if self.month is not None or self.day is not None or self.quarter is not None:
                # Allow but ignore for year type
                pass
        elif self.period_type == PeriodType.QUARTER:
            if self.quarter is None:
                raise ValueError("Quarter period requires quarter value")
        elif self.period_type == PeriodType.MONTH:
            if self.month is None:
                raise ValueError("Month period requires month value")
        elif self.period_type == PeriodType.DAY:
            if self.month is None or self.day is None:
                raise ValueError("Day period requires month and day values")
        return self

    # Factory methods (named to avoid conflict with field names)
    @classmethod
    def of_year(cls, year_val: int) -> Period:
        """Create a year period."""
        return cls(year=year_val, period_type=PeriodType.YEAR)

    @classmethod
    def of_month(cls, year_val: int, month_val: int) -> Period:
        """Create a month period."""
        return cls(year=year_val, month=month_val, period_type=PeriodType.MONTH)

    @classmethod
    def of_quarter(cls, year_val: int, quarter_val: int) -> Period:
        """Create a quarter period."""
        return cls(year=year_val, quarter=quarter_val, period_type=PeriodType.QUARTER)

    @classmethod
    def of_day(cls, year_val: int, month_val: int, day_val: int) -> Period:
        """Create a day period."""
        return cls(year=year_val, month=month_val, day=day_val, period_type=PeriodType.DAY)


    @classmethod
    def from_date(cls, d: date, period_type: PeriodType = PeriodType.DAY) -> Period:
        """Create period from date object."""
        if period_type == PeriodType.DAY:
            return cls.of_day(d.year, d.month, d.day)
        elif period_type == PeriodType.MONTH:
            return cls.of_month(d.year, d.month)
        elif period_type == PeriodType.QUARTER:
            q = (d.month - 1) // 3 + 1
            return cls.of_quarter(d.year, q)
        else:
            return cls.of_year(d.year)

    @classmethod
    def parse(cls, s: str) -> Period:
        """Parse period from string.

        Formats:
            - "2024" -> Year
            - "2024-03" -> Month
            - "2024-Q2" -> Quarter
            - "2024-03-15" -> Day
        """
        s = s.strip()

        # Year only
        if len(s) == 4 and s.isdigit():
            return cls.of_year(int(s))

        # Quarter
        if "-Q" in s.upper():
            parts = s.upper().split("-Q")
            return cls.of_quarter(int(parts[0]), int(parts[1]))

        # Month or Day
        parts = s.split("-")
        if len(parts) == 2:
            return cls.of_month(int(parts[0]), int(parts[1]))
        elif len(parts) == 3:
            return cls.of_day(int(parts[0]), int(parts[1]), int(parts[2]))

        raise ValueError(f"Cannot parse period: {s}")

    # Arithmetic
    def __add__(self, n: int) -> Period:
        """Add n units of this period's type."""
        if self.period_type == PeriodType.YEAR:
            return Period.of_year(self.year + n)
        elif self.period_type == PeriodType.QUARTER:
            total_quarters = (self.year * 4) + (self.quarter or 1) - 1 + n
            new_year = total_quarters // 4
            new_quarter = (total_quarters % 4) + 1
            return Period.of_quarter(new_year, new_quarter)
        elif self.period_type == PeriodType.MONTH:
            total_months = (self.year * 12) + (self.month or 1) - 1 + n
            new_year = total_months // 12
            new_month = (total_months % 12) + 1
            return Period.of_month(new_year, new_month)
        elif self.period_type == PeriodType.DAY:
            d = date(self.year, self.month or 1, self.day or 1)
            from datetime import timedelta

            new_date = d + timedelta(days=n)
            return Period.from_date(new_date, PeriodType.DAY)
        raise ValueError(f"Unknown period type: {self.period_type}")

    def __sub__(self, other: Period | int) -> int | Period:
        """Subtract periods (returns count) or int (returns period)."""
        if isinstance(other, int):
            return self + (-other)

        # Calculate difference in period units
        if self.period_type != other.period_type:
            raise ValueError("Cannot subtract periods of different types")

        if self.period_type == PeriodType.YEAR:
            return self.year - other.year
        elif self.period_type == PeriodType.QUARTER:
            return (self.year * 4 + (self.quarter or 1)) - (other.year * 4 + (other.quarter or 1))
        elif self.period_type == PeriodType.MONTH:
            return (self.year * 12 + (self.month or 1)) - (other.year * 12 + (other.month or 1))
        elif self.period_type == PeriodType.DAY:
            d1 = date(self.year, self.month or 1, self.day or 1)
            d2 = date(other.year, other.month or 1, other.day or 1)
            return (d1 - d2).days
        raise ValueError(f"Unknown period type: {self.period_type}")

    @classmethod
    def range(cls, start: Period, end: Period) -> Iterator[Period]:
        """Generate all periods from start to end (inclusive)."""
        if start.period_type != end.period_type:
            raise ValueError("Range requires same period types")

        current = start
        while current <= end:
            yield current
            current = current + 1

    # Containment
    def contains(self, other: Period) -> bool:
        """Check if this period contains another."""
        if not self.period_type.contains(other.period_type):
            if self.period_type == other.period_type:
                return self == other
            return False

        # Year contains month/quarter/day
        if self.period_type == PeriodType.YEAR:
            return self.year == other.year

        # Quarter contains month/day
        if self.period_type == PeriodType.QUARTER:
            if other.year != self.year:
                return False
            q_start = ((self.quarter or 1) - 1) * 3 + 1
            q_end = q_start + 2
            if other.period_type == PeriodType.MONTH:
                return q_start <= (other.month or 1) <= q_end
            if other.period_type == PeriodType.DAY:
                return q_start <= (other.month or 1) <= q_end

        # Month contains day
        if self.period_type == PeriodType.MONTH:
            return self.year == other.year and self.month == other.month

        return False

    def overlaps(self, other: Period) -> bool:
        """Check if periods overlap."""
        return self.contains(other) or other.contains(self)

    # Comparison
    def _to_ordinal(self) -> tuple:
        """Convert to ordinal for comparison."""
        if self.period_type == PeriodType.YEAR:
            return (self.year, 0, 0)
        elif self.period_type == PeriodType.QUARTER:
            return (self.year, (self.quarter or 1) * 3, 0)
        elif self.period_type == PeriodType.MONTH:
            return (self.year, self.month or 1, 0)
        elif self.period_type == PeriodType.DAY:
            return (self.year, self.month or 1, self.day or 1)
        return (self.year, 0, 0)

    def __lt__(self, other: Period) -> bool:
        return self._to_ordinal() < other._to_ordinal()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Period):
            return False
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
            and self.quarter == other.quarter
            and self.period_type == other.period_type
        )

    def __hash__(self) -> int:
        return hash((self.year, self.month, self.day, self.quarter, self.period_type))

    # String representation
    def __str__(self) -> str:
        if self.period_type == PeriodType.YEAR:
            return str(self.year)
        elif self.period_type == PeriodType.QUARTER:
            return f"{self.year}-Q{self.quarter}"
        elif self.period_type == PeriodType.MONTH:
            return f"{self.year}-{self.month:02d}"
        elif self.period_type == PeriodType.DAY:
            return f"{self.year}-{self.month:02d}-{self.day:02d}"
        return str(self.year)

    def __repr__(self) -> str:
        return f"Period({str(self)!r})"
