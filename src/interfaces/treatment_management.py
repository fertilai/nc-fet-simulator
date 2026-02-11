import datetime
from dataclasses import dataclass, field
from src.interfaces.enums import (
    CompareType,
    InstructionType,
    Weekday,
    OvulationPredictionDayPredictionWithMultipleClasses
)

from datetime import timedelta
from functools import cached_property
from typing import Tuple, Optional

import numpy as np

from src.interfaces.treatment import *
from abc import ABC, abstractmethod

@dataclass(kw_only=True)
class TreatmentInstructions:
    instructions: dict[int, list[InstructionType]] = field(default_factory=dict)
    cycle_missed: bool = False

    def add_instruction(self, day: int, instruction_type: InstructionType):
        if day in self.instructions:
            self.instructions[day].append(instruction_type)
        else:
            self.instructions[day] = [instruction_type]

    def add_instructions(self, day: int, instruction_types: list[InstructionType]):
        if day in self.instructions:
            self.instructions[day].extend(instruction_types)
        else:
            self.instructions[day] = instruction_types



@dataclass(kw_only=True)
class OvulationPredictionTreatmentResults:
    success_rate: float
    miss_rate: float
    average_test_count: float
    prediction_day_probabilities: dict[int, float]
    prediction_class_probabilities: dict[str, float]
    prediction_class_and_day_probabilities: dict[str, dict[int, float]]
    prediction_class_early_warning_probabilities: dict[str, float]
    transfer_on_unwanted_days: float = field(default_factory=float)
    ovulation_was_determined: float = field(default_factory=float)
    ovulation_on_unwanted_days_distribution: dict[str, float] = field(default_factory=dict)
    trigger_suggestion_rate: float = field(default_factory=float)
    relative_trigger_ovulation_day: dict[str, float] = field(default_factory=dict)

    @property
    def error_rate(self):
        return 1 - self.success_rate - self.miss_rate

    @property
    def error_and_miss_rate(self):
        return 1 - self.success_rate

    def update(self, probability: float, ovulation_prediction_treatment_results: "OvulationPredictionTreatmentResults") -> None:
        self.success_rate += probability * ovulation_prediction_treatment_results.success_rate
        self.miss_rate += probability * ovulation_prediction_treatment_results.miss_rate
        self.average_test_count += probability * ovulation_prediction_treatment_results.average_test_count
        self.transfer_on_unwanted_days += probability * ovulation_prediction_treatment_results.transfer_on_unwanted_days
        self.trigger_suggestion_rate += probability * ovulation_prediction_treatment_results.trigger_suggestion_rate
        self.ovulation_was_determined += probability * ovulation_prediction_treatment_results.ovulation_was_determined

        # update prediction_day_probabilities
        for prediction_day in ovulation_prediction_treatment_results.prediction_day_probabilities:
            if prediction_day not in self.prediction_day_probabilities:
                self.prediction_day_probabilities[prediction_day] = 0
            self.prediction_day_probabilities[prediction_day] += probability * ovulation_prediction_treatment_results.prediction_day_probabilities[prediction_day]

        # update prediction_class_probabilities
        for prediction_class in ovulation_prediction_treatment_results.prediction_class_probabilities:
            if prediction_class not in self.prediction_class_probabilities:
                self.prediction_class_probabilities[prediction_class] = 0
            self.prediction_class_probabilities[prediction_class] += probability * ovulation_prediction_treatment_results.prediction_class_probabilities[prediction_class]

        # update prediction_class_and_day_probabilities
        for prediction_class, prediction_days in ovulation_prediction_treatment_results.prediction_class_and_day_probabilities.items():
            if prediction_class not in self.prediction_class_and_day_probabilities:
                self.prediction_class_and_day_probabilities[prediction_class] = {}
            for prediction_day in prediction_days:
                if prediction_day not in self.prediction_class_and_day_probabilities[prediction_class]:
                    self.prediction_class_and_day_probabilities[prediction_class][prediction_day] = 0
                self.prediction_class_and_day_probabilities[prediction_class][prediction_day] += probability * ovulation_prediction_treatment_results.prediction_class_and_day_probabilities[prediction_class][prediction_day]

        # update relative_trigger_ovulation_day
        for relative_ovulation_day in ovulation_prediction_treatment_results.relative_trigger_ovulation_day:
            if relative_ovulation_day not in self.relative_trigger_ovulation_day:
                self.relative_trigger_ovulation_day[relative_ovulation_day] = 0
            self.relative_trigger_ovulation_day[relative_ovulation_day] += probability * ovulation_prediction_treatment_results.relative_trigger_ovulation_day[relative_ovulation_day]

        # update ovulation_on_unwanted_days_distribution
        for ovulation_day in ovulation_prediction_treatment_results.ovulation_on_unwanted_days_distribution:
            if ovulation_day not in self.ovulation_on_unwanted_days_distribution:
                self.ovulation_on_unwanted_days_distribution[ovulation_day] = 0
            self.ovulation_on_unwanted_days_distribution[ovulation_day] += probability * ovulation_prediction_treatment_results.ovulation_on_unwanted_days_distribution[ovulation_day]

        # update prediction_class_early_warning_probabilities
        for prediction_class in ovulation_prediction_treatment_results.prediction_class_early_warning_probabilities:
            if prediction_class not in self.prediction_class_early_warning_probabilities:
                self.prediction_class_early_warning_probabilities[prediction_class] = 0
            self.prediction_class_early_warning_probabilities[prediction_class] += probability * ovulation_prediction_treatment_results.prediction_class_early_warning_probabilities[prediction_class]

    @property
    def prediction_day_probabilities_sum(self) -> float:
        return sum(self.prediction_day_probabilities.values())

    @property
    def rounded_prediction_day_probabilities(self) -> dict[int, float]:
        return {day: round(probability, 2) for day, probability in self.prediction_day_probabilities.items() if round(probability, 2) > 0}

    @property
    def prediction_class_probabilities_sum(self) -> float:
        return sum(self.prediction_class_probabilities.values())

    @property
    def rounded_prediction_class_probabilities(self) -> dict[str, float]:
        return {class_: round(probability, 2) for class_, probability in self.prediction_class_probabilities.items() if round(probability, 2) > 0}

    @property
    def rounded_prediction_class_early_warning_probabilities(self) -> dict[str, float]:
        return {class_: round(probability / self.prediction_class_probabilities[class_], 2) for class_, probability in self.prediction_class_early_warning_probabilities.items() if round(probability, 2) > 0}

    def __str__(self) -> str:
        return f"success rate: {round(self.success_rate, 3)}, miss rate: {round(self.miss_rate, 3)}, transfer on unwanted days: {round(self.transfer_on_unwanted_days, 3)}, trigger suggestion rate: {round(self.trigger_suggestion_rate, 3)}, test count: {round(self.average_test_count, 2)},  ovulation was determined: {round(self.ovulation_was_determined, 3)},  ovulation on unwanted days distribution: {self.ovulation_on_unwanted_days_distribution}, prediction day probabilities: {self.rounded_prediction_day_probabilities},  prediction class probabilities: {self.rounded_prediction_class_probabilities},  relative trigger ovulation day: {self.relative_trigger_ovulation_day}, prediction class early warning probabilities: {self.rounded_prediction_class_early_warning_probabilities}"


@dataclass
class ModifiedFetUnwantedDays:
    embryo_age: int
    cycle_start_date: datetime.date

    clinic_closed_dates: list[datetime.date] = field(default_factory=list)
    clinic_minimal_staff_dates: list[datetime.date] = field(default_factory=list)

    trigger_triggers_ovulation_num_days = 2

    def get_ovulation_dates_from_transfer_dates(self, transfer_dates):
        ovulation_dates = []
        for transfer_date in transfer_dates:
            ovulation_dates.append(transfer_date - datetime.timedelta(days=self.embryo_age))
        return ovulation_dates

    def get_transfer_days_from_transfer_dates(self, transfer_dates):
        transfer_cycle_days = []
        for transfer_date in transfer_dates:
            transfer_cycle_day = (transfer_date - self.cycle_start_date).days + 1
            if transfer_cycle_day > 0:
                transfer_cycle_days.append(transfer_cycle_day)
        return transfer_cycle_days

    def get_ovulation_days_from_ovulation_dates(self, ovulation_dates):
        ovulation_cycle_days = []
        for ovulation_date in ovulation_dates:
            ovulation_cycle_day = (ovulation_date - self.cycle_start_date).days + 1
            if ovulation_cycle_day > 0:
                ovulation_cycle_days.append(ovulation_cycle_day)
        return ovulation_cycle_days

    def get_transfer_days_from_ovulation_days(self, ovulation_days):
        return [day + self.embryo_age for day in ovulation_days]

    @cached_property
    def num_consecutive_closed_days(self) -> Optional[int]:
        if self.clinic_closed_dates:
            distance_between_unwanted_days = (np.array(self.clinic_closed_dates)[1:] - np.array(self.clinic_closed_dates)[:-1])
            distance_between_unwanted_days_ints = np.array([d.days for d in distance_between_unwanted_days])
            if (distance_between_unwanted_days_ints == 1).any():
                indexs_one_day_from_other_unwanted_day = np.argwhere(distance_between_unwanted_days_ints == 1).ravel()
                if ((indexs_one_day_from_other_unwanted_day[1:] - indexs_one_day_from_other_unwanted_day[:-1]) == 1).any():
                    raise ValueError('More than 2 closed clinic days in a row')
                else:
                    return 2
            else:
                return 1
        else:
            return None

    @cached_property
    def ovulation_dates_that_lead_to_transfer_on_clinic_closed_days(self) -> list[datetime.date]:
        if self.clinic_closed_dates:
            unwanted_ovulation_dates = self.get_ovulation_dates_from_transfer_dates(self.clinic_closed_dates)
            return unwanted_ovulation_dates
        else:
            return []

    @cached_property
    def ovulation_days_that_lead_to_transfer_on_clinic_closed_days(self) -> list[int]:
        if self.ovulation_dates_that_lead_to_transfer_on_clinic_closed_days:
            unwanted_ovulation_days = self.get_ovulation_days_from_ovulation_dates(self.ovulation_dates_that_lead_to_transfer_on_clinic_closed_days)
            return unwanted_ovulation_days
        else:
            return []

    @cached_property
    def clinic_closed_days(self) -> list[int]:
        if self.clinic_closed_dates:
            clinic_closed_days = self.get_transfer_days_from_transfer_dates(self.clinic_closed_dates)
            return clinic_closed_days
        else:
            return []

    @cached_property
    def clinic_minimal_staff_days(self) -> list[int]:
        if self.clinic_minimal_staff_dates:
            clinic_minimal_staff_days = self.get_transfer_days_from_transfer_dates(self.clinic_minimal_staff_dates)
            return clinic_minimal_staff_days
        else:
            return []

    @cached_property
    def ovulation_days_that_leads_to_transfer_on_minimal_staff_day(self) -> list[int]:
        if self.clinic_minimal_staff_days:
            ovulation_days = [day - self.embryo_age for day in self.clinic_minimal_staff_days]
            return ovulation_days
        else:
            return []

    @cached_property
    def unwanted_trigger_days(self) -> list[int]:
        if self.ovulation_days_that_lead_to_transfer_on_clinic_closed_days:
            return [day - self.trigger_triggers_ovulation_num_days for day in self.ovulation_days_that_lead_to_transfer_on_clinic_closed_days]
        else:
            return []

    def get_all_last_possible_trigger_days(self, unwanted_ovulation_cycle_days):
        num_days_between_trigger_and_unwanted_ovulation = self.trigger_triggers_ovulation_num_days + 1
        all_last_trigger_days = unwanted_ovulation_cycle_days[:-1][unwanted_ovulation_cycle_days[1:] - unwanted_ovulation_cycle_days[:-1] == 1] - num_days_between_trigger_and_unwanted_ovulation
        if all_last_trigger_days.any():
            pass
        else:
            all_last_trigger_days = unwanted_ovulation_cycle_days - num_days_between_trigger_and_unwanted_ovulation
        all_last_trigger_days = all_last_trigger_days[all_last_trigger_days > 0]
        return all_last_trigger_days

    @cached_property
    def mandatory_trigger_test_day(self) -> np.ndarray:
        unwanted_ovulation_cycle_days = np.array(self.ovulation_days_that_lead_to_transfer_on_clinic_closed_days)
        minimal_staff_day_unwanted_ovulation_days = np.array(self.ovulation_days_that_leads_to_transfer_on_minimal_staff_day)

        if self.num_consecutive_closed_days == 1:
            if self.ovulation_days_that_leads_to_transfer_on_minimal_staff_day:
                all_mandatory_trigger_test_days = minimal_staff_day_unwanted_ovulation_days - self.trigger_triggers_ovulation_num_days - 1
            else:
                all_mandatory_trigger_test_days = unwanted_ovulation_cycle_days - self.trigger_triggers_ovulation_num_days - self.num_consecutive_closed_days

            return all_mandatory_trigger_test_days[all_mandatory_trigger_test_days > 0]

        elif self.num_consecutive_closed_days == 2:
            all_mandatory_trigger_test_days = self.get_all_last_possible_trigger_days(unwanted_ovulation_cycle_days)
            return all_mandatory_trigger_test_days
        else:
            return np.array([])

    @cached_property
    def optional_early_trigger_test_days(self) -> np.ndarray:
        if len(self.mandatory_trigger_test_day):
            optional_early_trigger_test_days = self.mandatory_trigger_test_day - 1
            return optional_early_trigger_test_days[optional_early_trigger_test_days > 0]
        else:
            return np.array([])

    @cached_property
    def optional_late_trigger_test_day(self) -> np.ndarray:
        if self.clinic_minimal_staff_dates:
            return self.mandatory_trigger_test_day + 1
        else:
            return np.array([])

    @cached_property
    def all_possible_trigger_days(self):
        return np.sort(np.hstack([self.optional_early_trigger_test_days, self.mandatory_trigger_test_day, self.optional_late_trigger_test_day]))


@dataclass
class PossibleTreatmentResult:
    year: int
    month: int
    probability: float
    initial_test_day: int
    embryo_age: int
    true_natural_ovulation_day: int
    first_cycle_day: datetime.date
    relative_test_day: int
    trigger_was_suggested: bool
    ovulation_was_missed: bool
    test_days: list

    unwanted_ovulation_days: list
    clinic_closed_days: list

    mandatory_trigger_test_day: np.ndarray
    optional_early_trigger_test_days: np.ndarray
    optional_late_trigger_test_day: np.ndarray
    all_possible_trigger_days: np.ndarray

    there_are_closed_days: bool
    there_are_minimal_staff_days: bool

    all_previous_predictions: list[OvulationPredictionDayPredictionWithMultipleClasses] = field(default_factory=list)
    previous_greater_than_probability: list[float] = field(default_factory=list)

    prediction: OvulationPredictionDayPredictionWithMultipleClasses = None
    follicle_size_group: CompareType = None

    predicted_ovulation_day_relative_to_true_ovulation: int = None
    predicted_transfer_day: int = None
    relative_transfer_day: int = None
    relative_trigger_suggestion_day: int = None

    trigger_triggers_ovulation_num_days = 2

    def get_date_from_cycle_day(self, cycle_day: int):
        return self.first_cycle_day + timedelta(days=int(cycle_day) - 1)

    @classmethod
    def get_weekday_from_date(cls, date_: datetime.date) -> Weekday:
        weekday_int = date_.weekday() + 1
        return Weekday(weekday_int)

    @cached_property
    def prediction_name(self) -> Optional[str]:
        if self.prediction:
            return self.prediction.name
        return None

    @cached_property
    def target_transfer_day(self) -> int:
        return self.true_natural_ovulation_day + self.embryo_age

    @cached_property
    def true_transfer_date(self) -> Optional[datetime.date]:
        if self.target_transfer_day:
            return self.get_date_from_cycle_day(self.target_transfer_day)
        else:
            return None

    @cached_property
    def true_transfer_weekday(self) -> Optional[Weekday]:
        if self.true_transfer_date:
            return self.get_weekday_from_date(self.true_transfer_date)
        else:
            return None

    @cached_property
    def test_day(self) -> int:
        return self.test_days[-1]

    @cached_property
    def test_date(self) -> datetime.date:
        return self.get_date_from_cycle_day(self.test_day)

    @cached_property
    def test_weekday(self) -> Weekday:
        return self.get_weekday_from_date(self.test_date)

    def get_last_test_day_from_list_of_possible_trigger_days(self, possible_trigger_test_days):
        all_tests_on_possible_trigger_days = np.intersect1d(np.array(self.test_days), np.array(possible_trigger_test_days))
        if len(all_tests_on_possible_trigger_days):
            last_test_on_possible_trigger_day = np.max(all_tests_on_possible_trigger_days)
            return last_test_on_possible_trigger_day
        else:
            return None

    def prediction_on_possible_trigger_day(self, possible_trigger_day) -> Optional[OvulationPredictionDayPredictionWithMultipleClasses]:
        if self.all_previous_predictions:
            if possible_trigger_day:
                if len(self.test_days):
                    if self.test_days[-1] == possible_trigger_day:
                        return self.prediction
                    index_of_test_on_last_possible_trigger_day = np.argwhere(self.test_days[:-1] == possible_trigger_day).ravel()[0]
                    return self.all_previous_predictions[index_of_test_on_last_possible_trigger_day]
                else:
                    return None
            else:
                return None
        else:
            return None

    @cached_property
    def last_test_on_possible_trigger_day(self) -> Optional[int]:
        return self.get_last_test_day_from_list_of_possible_trigger_days(self.all_possible_trigger_days)

    @cached_property
    def last_test_on_possible_trigger_date(self) -> Optional[datetime.date]:
        if self.last_test_on_possible_trigger_day:
            return self.get_date_from_cycle_day(self.last_test_on_possible_trigger_day)
        else:
            return None

    @cached_property
    def last_test_on_possible_trigger_weekday(self) -> Optional[Weekday]:
        if self.last_test_on_possible_trigger_date:
            return self.get_weekday_from_date(self.last_test_on_possible_trigger_date)
        else:
            return None

    @cached_property
    def prediction_on_last_test_on_possible_trigger_day(self) -> Optional[OvulationPredictionDayPredictionWithMultipleClasses]:
        if self.last_test_on_possible_trigger_day:
            return self.prediction_on_possible_trigger_day(self.last_test_on_possible_trigger_day)
        else:
            return None

    @cached_property
    def prediction_name_on_last_test_on_possible_trigger_day(self) -> Optional[str]:
        if self.prediction_on_last_test_on_possible_trigger_day:
            return self.prediction_on_last_test_on_possible_trigger_day.name
        else:
            return None

    @cached_property
    def last_test_on_mandatory_possible_trigger_day(self) -> Optional[int]:
        return self.get_last_test_day_from_list_of_possible_trigger_days(self.mandatory_trigger_test_day)

    @cached_property
    def last_test_on_mandatory_possible_trigger_date(self) -> Optional[datetime.date]:
        if self.last_test_on_mandatory_possible_trigger_day:
            return self.get_date_from_cycle_day(self.last_test_on_mandatory_possible_trigger_day)
        else:
            return None

    @cached_property
    def last_test_on_mandatory_possible_trigger_weekday(self) -> Optional[Weekday]:
        if self.last_test_on_mandatory_possible_trigger_date:
            return self.get_weekday_from_date(self.last_test_on_mandatory_possible_trigger_date)
        else:
            return None

    @cached_property
    def prediction_on_last_mandatory_test_on_possible_trigger_day(self) -> Optional[OvulationPredictionDayPredictionWithMultipleClasses]:
        if self.last_test_on_mandatory_possible_trigger_day:
            return self.prediction_on_possible_trigger_day(self.last_test_on_mandatory_possible_trigger_day)
        else:
            return None

    @cached_property
    def prediction_name_on_last_mandatory_test_on_possible_trigger_day(self) -> Optional[str]:
        if self.prediction_on_last_mandatory_test_on_possible_trigger_day:
            return self.prediction_on_last_mandatory_test_on_possible_trigger_day.name
        else:
            return None

    @cached_property
    def last_test_on_optional_early_trigger_test_day(self) -> Optional[int]:
        return self.get_last_test_day_from_list_of_possible_trigger_days(self.optional_early_trigger_test_days)

    @cached_property
    def last_test_on_optional_early_trigger_test_date(self) -> Optional[datetime.date]:
        last_test_on_optional_early_trigger_test_day = self.last_test_on_optional_early_trigger_test_day
        if last_test_on_optional_early_trigger_test_day:
            return self.get_date_from_cycle_day(last_test_on_optional_early_trigger_test_day)
        else:
            return None

    @cached_property
    def last_test_on_optional_early_trigger_test_weekday(self) -> Optional[Weekday]:
        if self.last_test_on_optional_early_trigger_test_date:
            return self.get_weekday_from_date(self.last_test_on_optional_early_trigger_test_date)
        else:
            return None

    @cached_property
    def prediction_on_last_optional_early_test_on_possible_trigger_day(self) -> Optional[OvulationPredictionDayPredictionWithMultipleClasses]:
        if self.last_test_on_optional_early_trigger_test_day:
            return self.prediction_on_possible_trigger_day(self.last_test_on_optional_early_trigger_test_day)
        else:
            return None

    @cached_property
    def prediction_name_on_last_optional_early_test_on_possible_trigger_day(self) -> Optional[str]:
        if self.prediction_on_last_optional_early_test_on_possible_trigger_day:
            return self.prediction_on_last_optional_early_test_on_possible_trigger_day.name
        else:
            return None

    @cached_property
    def last_test_on_optional_late_trigger_test_day(self) -> Optional[int]:
        return self.get_last_test_day_from_list_of_possible_trigger_days(self.optional_late_trigger_test_day)

    @cached_property
    def last_test_on_optional_late_trigger_test_date(self) -> Optional[datetime.date]:
        if self.last_test_on_optional_late_trigger_test_day:
            return self.get_date_from_cycle_day(self.last_test_on_optional_late_trigger_test_day)
        else:
            return None

    @cached_property
    def last_test_on_optional_late_trigger_test_weekday(self) -> Optional[Weekday]:
        if self.last_test_on_optional_late_trigger_test_date:
            return self.get_weekday_from_date(self.last_test_on_optional_late_trigger_test_date)
        else:
            return None

    @cached_property
    def prediction_on_last_optional_late_test_on_possible_trigger_day(self) -> Optional[OvulationPredictionDayPredictionWithMultipleClasses]:
        if self.last_test_on_optional_late_trigger_test_day:
            return self.prediction_on_possible_trigger_day(self.last_test_on_optional_late_trigger_test_day)
        else:
            return None

    @cached_property
    def prediction_name_on_last_optional_late_test_on_possible_trigger_day(self) -> Optional[str]:
        if self.prediction_on_last_optional_late_test_on_possible_trigger_day:
            return self.prediction_on_last_optional_late_test_on_possible_trigger_day.name
        else:
            return None

    @cached_property
    def trigger_was_needed(self) -> bool:
        if self.target_transfer_day in self.clinic_closed_days:
            return True
        else:
            return False

    @cached_property
    def trigger_day(self) -> Optional[int]:
        if self.trigger_was_suggested:
            return self.true_natural_ovulation_day + self.relative_trigger_suggestion_day
        else:
            return None

    @cached_property
    def trigger_date(self) -> Optional[datetime.date]:
        if self.trigger_day:
            return self.get_date_from_cycle_day(self.trigger_day)
        else:
            return None

    @cached_property
    def trigger_weekday(self) -> Optional[Weekday]:
        if self.trigger_date:
            return self.get_weekday_from_date(self.trigger_date)
        else:
            return None

    @cached_property
    def predicted_transfer_date(self) -> Optional[datetime.date]:
        if self.predicted_transfer_day:
            return self.get_date_from_cycle_day(self.predicted_transfer_day)
        else:
            return None

    @cached_property
    def predicted_transfer_weekday(self) -> Optional[Weekday]:
        if self.predicted_transfer_date:
            return self.get_weekday_from_date(self.predicted_transfer_date)
        else:
            return None

    @cached_property
    def predicted_transfer_weekday_int(self) -> Optional[int]:
        if self.predicted_transfer_weekday:
            return self.predicted_transfer_weekday.value
        else:
            return None

    @cached_property
    def predicted_transfer_weekday_name(self) -> Optional[str]:
        if self.predicted_transfer_weekday:
            return self.predicted_transfer_weekday.name
        else:
            return None

    @cached_property
    def predicted_ovulation_day(self) -> Optional[int]:
        if self.predicted_transfer_day:
            return self.predicted_transfer_day - self.embryo_age
        else:
            return None

    @cached_property
    def previous_test_day(self) -> Optional[int]:
        if len(self.test_days) >= 2:
            return self.test_days[-2]
        else:
            return None

    @cached_property
    def days_from_previous_test_to_last_test(self) -> Optional[int]:
        if self.previous_test_day:
            return self.test_days[-1] - self.previous_test_day
        else:
            return None

    @cached_property
    def previous_distance_from_true_ovulation(self) -> Optional[int]:
        if self.all_previous_distance_from_true_ovulation:
            return self.all_previous_distance_from_true_ovulation[-1]
        else:
            return None

    @cached_property
    def previous_prediction(self) -> Optional[OvulationPredictionDayPredictionWithMultipleClasses]:
        if self.all_previous_predictions:
            return self.all_previous_predictions[-1]
        else:
            return None

    @cached_property
    def previous_prediction_name(self) -> Optional[str]:
        if self.all_previous_predictions_names:
            return self.all_previous_predictions_names[-1]
        else:
            return None

    @cached_property
    def all_previous_distance_from_true_ovulation(self) -> list[int]:
        prev_dist_from_ovulation = []
        for test_day in self.test_days:
            prev_dist_from_ovulation.append(test_day - self.true_natural_ovulation_day)
        return prev_dist_from_ovulation[:-1]

    @cached_property
    def all_previous_predictions_names(self) -> Optional[list[str]]:
        if self.all_previous_predictions:
            return [predictions.name for predictions in self.all_previous_predictions]
        else:
            return None

    @cached_property
    def success_for_treatment_by_follicle_size(self) -> bool:
        if self.relative_trigger_suggestion_day is None:
            return False
        else:
            if self.relative_trigger_suggestion_day > 0:
                return False
            else:
                return True

    @cached_property
    def success(self):
        if self.trigger_was_suggested:
            final_transfer_day = self.trigger_day + self.trigger_triggers_ovulation_num_days + self.embryo_age
        else:
            final_transfer_day = self.target_transfer_day

        if self.predicted_transfer_day == final_transfer_day:
            return True
        else:
            return False

    @cached_property
    def success_trigger(self):
        if self.trigger_was_suggested and self._has_success(
                final_transfer_day=self.trigger_day + self.trigger_triggers_ovulation_num_days + self.embryo_age):
            return True
        return False

    @cached_property
    def error_trigger(self):
        if self.trigger_was_suggested and self._has_error(
                final_transfer_day=self.trigger_day + self.trigger_triggers_ovulation_num_days + self.embryo_age):
            return True
        return False

    @cached_property
    def missed_trigger(self):
        if self.trigger_was_suggested and self._has_missed():
            return True
        return False

    @cached_property
    def success_natural(self):
        if not self.trigger_was_suggested and self._has_success(final_transfer_day=self.target_transfer_day):
            return True
        return False

    @cached_property
    def error_natural(self):
        if not self.trigger_was_suggested and self._has_error(final_transfer_day=self.target_transfer_day):
            return True
        return False

    @cached_property
    def missed_natural(self):
        if not self.trigger_was_suggested and self._has_missed():
            return True
        return False

    @cached_property
    def missed_natural_clinic_closure_day(self):
        if not self.trigger_was_suggested and self.transfer_on_unwanted_day:
            return True
        return False

    @cached_property
    def missed_natural_inaccurate_ovulation_prediction(self):
        if not self.trigger_was_suggested and self.ovulation_was_missed:
            return True
        return False

    def _has_success(self, final_transfer_day):
        # Success => Transfer on TargetTransfer AND Transfer not on closed day
        if self.predicted_transfer_day == final_transfer_day and not self.transfer_on_unwanted_day:
            return True
        return False

    def _has_error(self, final_transfer_day):
        # Error => no declared miss cycle and (Transfer not on TargetTransfer AND Transfer not on closed day)
        if not self.ovulation_was_missed:
            if self.predicted_transfer_day != final_transfer_day and not self.transfer_on_unwanted_day:
                return True
        return False

    def _has_missed(self):
        # Missed => Transfer on closed day OR declared miss cycle
        if self.transfer_on_unwanted_day or self.ovulation_was_missed:
            return True
        return False

    @cached_property
    def num_tests(self) -> int:
        return len(self.test_days)

    @cached_property
    def transfer_on_unwanted_day(self) -> bool:
        if self.predicted_transfer_day in self.clinic_closed_days:
            return True
        else:
            return False

    @cached_property
    def num_days_ovulation_was_moved_back(self) -> Optional[int]:
        if self.trigger_was_suggested:
            triggered_ovulation_day = self.trigger_day + 2
            return self.true_natural_ovulation_day - triggered_ovulation_day

        else:
            return None

    @cached_property
    def trigger_but_transfer_on_unwanted_day(self) -> bool:
        if self.trigger_was_suggested and self.predicted_transfer_day in self.clinic_closed_days:
            return True
        else:
            return False

class FetTreatmentManager(ABC):
    default_initial_test_day: int

    @abstractmethod
    def get_initial_test_day(self, cycle_start_date: datetime.date, embryo_age: int,
                             clinic_closed_dates: list[datetime.date] = None,
                             clinic_minimal_staff_dates: list[datetime.date] = None) -> int:
        pass

    @abstractmethod
    def get_treatment_cycle_instruction_from_prediction_class_options(self, prediction_class_options,
                                                                      current_test_day: int,
                                                                      cycle_start_date: datetime.date,
                                                                      embryo_age: int = None,
                                                                      clinic_closed_dates: list[datetime.date] = None,
                                                                      clinic_minimal_staff_dates: list[
                                                                          datetime.date] = None
                                                                      ) -> TreatmentInstructions:
        pass

    @abstractmethod
    def get_ovulation_predicted_cycle_day(self,
                                          relative_predicted_ovulation_day: OvulationPredictionDayPredictionWithMultipleClasses,
                                          test_day: int) -> tuple[int] | tuple[int, int]:
        pass

    @abstractmethod
    def choose_ovulation_prediction_day(self, class_options_predicted_ovulation_days: tuple,
                                        unwanted_ovulation_days: list[int]) -> int:
        pass
