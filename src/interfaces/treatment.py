from __future__ import annotations
from dataclasses import dataclass
import datetime

@dataclass
class TreatmentCycle:
    id: str | int | None = None
    cycle_id: str | int | None = None
    menstruation_start_date: datetime.date | None = None
    blood_tests_by_cycle_day: dict = None
