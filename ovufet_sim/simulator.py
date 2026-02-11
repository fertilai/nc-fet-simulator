from __future__ import annotations

"""Core simulator implementation.

This repository is a minimal, self-contained slice used to reproduce the paper
results from:
- an evaluation Excel file (model outputs per test),
- a precomputed confusion-matrix dictionary (CM) produced from the original codebase.

The simulator itself is deterministic (probability-weighted recursion, not Monte Carlo).
"""

from calendar import monthrange
from itertools import islice
import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.algorithms.ovulation_prediction import (
    RelativeOvulationDay,
    OvulationPredictionDayPredictionWithMultipleClasses,
    OvulationPredictionDayTargetClass,
)
from src.interfaces.treatment_management import (
    TreatmentInstructions,
    OvulationPredictionTreatmentResults,
    ModifiedFetUnwantedDays,
    PossibleTreatmentResult,
)
from src.interfaces.treatment import *
from src.interfaces.enums import *


class NaturalCycleStatistics:
    default_min_allowed_ovulation_day = 10
    default_max_allowed_ovulation_day = 21
    min_relative_ovulation_day = -14
    max_relative_ovulation_day = 14

    def __init__(self, treatment_cycles: list[TreatmentCycle], min_allowed_ovulation_day: int = None,
                 max_allowed_ovulation_day: int = None, ovulation_day_probabilities: dict | None = None):

        print(f"Generating statistics using {len(treatment_cycles)} treatment cycles.")
        self._provided_ovulation_day_probabilities = ovulation_day_probabilities

        self.set_min_max_allowed_ovulation_days(min_allowed_ovulation_day, max_allowed_ovulation_day)

        self.data = {}
        self.cycles_per_relative_day = {}

        self._generate_ovulation_day_statistics()

        print(f"Generated statistics.")

    def set_min_max_allowed_ovulation_days(self, min_allowed_ovulation_day, max_allowed_ovulation_day):

        if min_allowed_ovulation_day:
            self.min_allowed_ovulation_day = min_allowed_ovulation_day
        else:
            self.min_allowed_ovulation_day = self.default_min_allowed_ovulation_day

        if max_allowed_ovulation_day:
            self.max_allowed_ovulation_day = max_allowed_ovulation_day
        else:
            self.max_allowed_ovulation_day = self.default_max_allowed_ovulation_day

    def _generate_ovulation_day_statistics(self):
        self.ovulation_day_counts = {day: 0 for day in
                                     range(self.min_allowed_ovulation_day, self.max_allowed_ovulation_day + 1)}

        if getattr(self, "_provided_ovulation_day_probabilities", None):
            self.ovulation_day_probabilities = dict(self._provided_ovulation_day_probabilities)
        else:
            self.ovulation_day_probabilities = {}


class OvulationPredictionTreatmentResultsCalculator:
    embryo_age = 5
    year = 2023
    month = 1
    ONE_TEST_BEFORE_OR_EQUAL_DAY_10_CM = 'one_test_before_or_equal_day_10_cm'
    ONE_TEST_AFTER_DAY_10_CM = 'one_test_after_day_10_cm'
    TWO_TESTS_CM = 'two_tests_cm'

    def __init__(self, model, first_test_day_name: str, second_test_day_name: str,
                 treatment_cycles: list[TreatmentCycle],
                 treatment_manager, clinic_closed_weekdays: list[Weekday] = None,
                 clinic_minimal_staff_weekdays: list[Weekday] = None,
                 randomize_confusion_matrices: bool = False, print_cm: bool = False,
                 predicted_class_probabilities: dict | None = None,
                 ovulation_day_probabilities: dict | None = None):
        self.model = model
        self.statistics = NaturalCycleStatistics(
            treatment_cycles,
            ovulation_day_probabilities=ovulation_day_probabilities,
        )
        self.first_test_day_name = first_test_day_name
        self.second_test_day_name = second_test_day_name
        self.treatment_manager = treatment_manager

        self.randomize_confusion_matrices = randomize_confusion_matrices
        self.print_cm = print_cm
        self.clinic_closed_dates = self.calculate_unwanted_transfer_dates(clinic_closed_weekdays)
        self.clinic_minimal_staff_dates = self.calculate_unwanted_transfer_dates(clinic_minimal_staff_weekdays)

        if clinic_closed_weekdays:
            self.there_are_closed_days = True
        else:
            self.there_are_closed_days = False

        if clinic_minimal_staff_weekdays:
            self.there_are_minimal_staff_days = True
        else:
            self.there_are_minimal_staff_days = False

        self.predicted_class_probabilities = predicted_class_probabilities or {}
        self.relative_prediction_days_to_treatment_results = {}

        self._generate_target_classes()
        if not self.predicted_class_probabilities:
            raise ValueError('predicted_class_probabilities is required (provide a precomputed confusion matrix).')

    @property
    def model_parameters(self):
        return self.model.model_parameters

    @property
    def model_target_classes_in_order(self) -> list[OvulationPredictionDayTargetClass]:
        return [target for target in RelativeOvulationDay.convert_enum_label_to_int.keys()]

    @property
    def target_class_to_confusion_matrix_index(self) -> dict[str, int]:
        return {self.target_classes[i]: i for i in range(len(self.target_classes))}

    @classmethod
    def get_target_class(cls, relative_prediction_day: int, raise_error=True) -> Optional[str]:
        low_target = RelativeOvulationDay.convert_enum_label_to_int[
            min(OvulationPredictionDayPredictionWithMultipleClasses)]
        high_target = RelativeOvulationDay.convert_enum_label_to_int[
            max(OvulationPredictionDayPredictionWithMultipleClasses)]
        if relative_prediction_day is None:
            print(f"Error: received null relative prediction day - cannot generate target class")
            return None
        if low_target is not None and relative_prediction_day <= low_target:
            target_value = RelativeOvulationDay.convert_int_to_str[low_target]
        elif high_target is not None and relative_prediction_day >= high_target:
            target_value = RelativeOvulationDay.convert_int_to_str[high_target]
        elif relative_prediction_day < high_target or relative_prediction_day > low_target:
            target_value = RelativeOvulationDay.get_relative_day_from_int_to_str(relative_prediction_day)
        else:
            if raise_error:
                raise Exception(
                    f"Error: cannot map relative prediction day: {relative_prediction_day} cannot generate target class")
            else:
                target_value = None
        return target_value

    def calculate_unwanted_transfer_dates(self, weekdays):
        if weekdays:
            unwanted_transfer_dates = []
            num_days_in_month = monthrange(self.year, self.month)[1]
            month_days = np.arange(1, num_days_in_month)
            for day in month_days:
                weekday_value = datetime.date(self.year, self.month, day).weekday() + 1
                weekday = Weekday(weekday_value)
                if weekday in weekdays:
                    unwanted_transfer_dates.append(datetime.date(self.year, self.month, day))
        else:
            unwanted_transfer_dates = []
        return unwanted_transfer_dates

    @classmethod
    def get_transfer_day(cls, next_instruction: TreatmentInstructions) -> Optional[int]:
        # if Natural fet and not modified - there is no InstructionType.Transfer and transfer day is None
        transfer_day = None
        for day, instruction_type in next_instruction.instructions.items():
            if InstructionType.TRANSFER in instruction_type:
                transfer_day = day
        return transfer_day

    @classmethod
    def get_trigger_day(cls, next_instruction):
        trigger_day = None
        for day, instruction_type in next_instruction.instructions.items():
            if InstructionType.TRIGGER in instruction_type:
                trigger_day = day
        return trigger_day

    @classmethod
    def get_treatment_results_df_new(cls, treatment_results: list[PossibleTreatmentResult],
                                     get_data_for_full_analysis: bool = False, chunk_size: int = 10000) -> pd.DataFrame:
        def chunked_iterable(iterable, size):
            it = iter(iterable)
            while True:
                chunk = list(islice(it, size))
                if not chunk:
                    break
                yield chunk

        def process_chunk(chunk):
            return [{
                'probability': res.probability,
                'success': res.success,
                'trigger_was_suggested': res.trigger_was_suggested, 'trigger_was_needed': res.trigger_was_needed,
                'transfer_on_unwanted_day': res.transfer_on_unwanted_day,
                'ovulation_was_missed': res.ovulation_was_missed, 'num_tests': res.num_tests,
                'trigger_but_transfer_on_unwanted_day': res.trigger_but_transfer_on_unwanted_day,
                'first_cycle_day': res.first_cycle_day, 'initial_test_day': res.initial_test_day,
                'true_ovulation_day': res.true_natural_ovulation_day, 'test_day': res.test_day,
                'test_date': res.test_date, 'test_weekday': res.test_weekday,
                'last_test_on_possible_trigger_day': res.last_test_on_possible_trigger_day,
                'last_test_on_possible_trigger_date': res.last_test_on_possible_trigger_date,
                'last_test_on_possible_trigger_weekday': res.last_test_on_possible_trigger_weekday,
                'prediction_on_last_test_on_possible_trigger_day': res.prediction_on_last_test_on_possible_trigger_day,
                'prediction_name_on_last_test_on_possible_trigger_day': res.prediction_name_on_last_test_on_possible_trigger_day,
                'last_test_on_mandatory_possible_trigger_day': res.last_test_on_mandatory_possible_trigger_day,
                'last_test_on_mandatory_possible_trigger_date': res.last_test_on_mandatory_possible_trigger_date,
                'last_test_on_mandatory_possible_trigger_weekday': res.last_test_on_mandatory_possible_trigger_weekday,
                'prediction_on_last_mandatory_test_on_possible_trigger_day': res.prediction_on_last_mandatory_test_on_possible_trigger_day,
                'prediction_name_on_last_mandatory_test_on_possible_trigger_day': res.prediction_name_on_last_mandatory_test_on_possible_trigger_day,
                'last_test_on_optional_early_trigger_test_day': res.last_test_on_optional_early_trigger_test_day,
                'last_test_on_optional_early_trigger_test_date': res.last_test_on_optional_early_trigger_test_date,
                'last_test_on_optional_early_trigger_test_weekday': res.last_test_on_optional_early_trigger_test_weekday,
                'prediction_on_last_optional_early_test_on_possible_trigger_day': res.prediction_on_last_optional_early_test_on_possible_trigger_day,
                'prediction_name_on_last_optional_early_test_on_possible_trigger_day': res.prediction_name_on_last_optional_early_test_on_possible_trigger_day,
                'last_test_on_optional_late_trigger_test_day': res.last_test_on_optional_late_trigger_test_day,
                'last_test_on_optional_late_trigger_test_date': res.last_test_on_optional_late_trigger_test_date,
                'last_test_on_optional_late_trigger_test_weekday': res.last_test_on_optional_late_trigger_test_weekday,
                'prediction_on_last_optional_late_test_on_possible_trigger_day': res.prediction_on_last_optional_late_test_on_possible_trigger_day,
                'prediction_name_on_last_optional_late_test_on_possible_trigger_day': res.prediction_name_on_last_optional_late_test_on_possible_trigger_day,
                'true_transfer_day': res.target_transfer_day, 'true_transfer_date': res.true_transfer_date,
                'true_transfer_weekday': res.true_transfer_weekday,
                'predicted_ovulation_day': res.predicted_ovulation_day,
                'transfer_day': res.predicted_transfer_day, 'transfer_date': res.predicted_transfer_date,
                'transfer_weekday': res.predicted_transfer_weekday,
                'transfer_weekday_name': res.predicted_transfer_weekday_name,
                'transfer_weekday_int': res.predicted_transfer_weekday_int, 'trigger_day': res.trigger_day,
                'trigger_date': res.trigger_date, 'trigger_weekday': res.trigger_weekday,
                'num_days_ovulation_was_moved_back': res.num_days_ovulation_was_moved_back,
                'relative_trigger_suggestion_day': res.relative_trigger_suggestion_day,
                'relative_transfer_day': res.relative_transfer_day,
                'unwanted_ovulation_days': res.unwanted_ovulation_days, 'clinic_closed_days': res.clinic_closed_days,
                'mandatory_trigger_test_day': res.mandatory_trigger_test_day,
                'optional_early_trigger_test_days': res.optional_early_trigger_test_days,
                'optional_late_trigger_test_day': res.optional_late_trigger_test_day,
                'all_possible_trigger_days': res.all_possible_trigger_days,
                'all_previous_distance_from_true_ovulation': res.all_previous_distance_from_true_ovulation,
                'previous_distance_from_true_ovulation': res.previous_distance_from_true_ovulation,
                'previous_prediction': res.previous_prediction,
                'previous_prediction_name': res.previous_prediction_name,
                'success_trigger': res.success_trigger,
                'success_natural': res.success_natural,
                'error_trigger': res.error_trigger,
                'error_natural': res.error_natural,
                'missed_trigger': res.missed_trigger,
                'missed_natural': res.missed_natural,
                'missed_natural_clinic_closure_day': res.missed_natural_clinic_closure_day,
                'missed_natural_inaccurate_ovulation_prediction': res.missed_natural_inaccurate_ovulation_prediction,
            } for res in chunk]

        if not get_data_for_full_analysis:
            data = [{
                'probability': res.probability,
                'success': res.success,
                'trigger_was_suggested': res.trigger_was_suggested,
                'trigger_was_needed': res.trigger_was_needed,
                'transfer_on_unwanted_day': res.transfer_on_unwanted_day,
                'ovulation_was_missed': res.ovulation_was_missed,
                'num_tests': res.num_tests,
                'transfer_weekday': res.predicted_transfer_weekday,
                'num_days_ovulation_was_moved_back': res.num_days_ovulation_was_moved_back,
                'success_trigger': res.success_trigger,
                'success_natural': res.success_natural,
                'error_trigger': res.error_trigger,
                'error_natural': res.error_natural,
                'missed_trigger': res.missed_trigger,
                'missed_natural': res.missed_natural,
                'missed_natural_clinic_closure_day': res.missed_natural_clinic_closure_day,
                'missed_natural_inaccurate_ovulation_prediction': res.missed_natural_inaccurate_ovulation_prediction,
            } for res in treatment_results]
            return pd.DataFrame(data)
        else:
            result_data = []
            num_chunks = len(treatment_results) // chunk_size + 1
            for chunk in chunked_iterable(treatment_results, chunk_size):
                result_data.extend(process_chunk(chunk))
            return pd.DataFrame(result_data)

    @classmethod
    def get_treatment_results_df(cls, treatment_results: list[PossibleTreatmentResult],
                                 get_data_for_full_analysis: bool = False) -> pd.DataFrame:
        res_dict = {}
        if not get_data_for_full_analysis:
            for i, res in enumerate(treatment_results):
                res_dict[i] = {'probability': res.probability, 'success': res.success,
                               'trigger_was_suggested': res.trigger_was_suggested,
                               'trigger_was_needed': res.trigger_was_needed,
                               'transfer_on_unwanted_day': res.transfer_on_unwanted_day,
                               'ovulation_was_missed': res.ovulation_was_missed, 'num_tests': res.num_tests,
                               'transfer_weekday': res.predicted_transfer_weekday,
                               'num_days_ovulation_was_moved_back': res.num_days_ovulation_was_moved_back}
        else:
            for i, res in enumerate(treatment_results):
                res_dict[i] = {'probability': res.probability, 'success': res.success,
                               'trigger_was_suggested': res.trigger_was_suggested,
                               'trigger_was_needed': res.trigger_was_needed,
                               'transfer_on_unwanted_day': res.transfer_on_unwanted_day,
                               'ovulation_was_missed': res.ovulation_was_missed, 'num_tests': res.num_tests,
                               'trigger_but_transfer_on_unwanted_day': res.trigger_but_transfer_on_unwanted_day,
                               'first_cycle_day': res.first_cycle_day, 'initial_test_day': res.initial_test_day,
                               'true_ovulation_day': res.true_natural_ovulation_day,
                               'test_day': res.test_day, 'test_date': res.test_date, 'test_weekday': res.test_weekday,

                               'last_test_on_possible_trigger_day': res.last_test_on_possible_trigger_day,
                               'last_test_on_possible_trigger_date': res.last_test_on_possible_trigger_date,
                               'last_test_on_possible_trigger_weekday': res.last_test_on_possible_trigger_weekday,
                               'prediction_on_last_test_on_possible_trigger_day': res.prediction_on_last_test_on_possible_trigger_day,
                               'prediction_name_on_last_test_on_possible_trigger_day': res.prediction_name_on_last_test_on_possible_trigger_day,

                               'last_test_on_mandatory_possible_trigger_day': res.last_test_on_mandatory_possible_trigger_day,
                               'last_test_on_mandatory_possible_trigger_date': res.last_test_on_mandatory_possible_trigger_date,
                               'last_test_on_mandatory_possible_trigger_weekday': res.last_test_on_mandatory_possible_trigger_weekday,
                               'prediction_on_last_mandatory_test_on_possible_trigger_day': res.prediction_on_last_mandatory_test_on_possible_trigger_day,
                               'prediction_name_on_last_mandatory_test_on_possible_trigger_day': res.prediction_name_on_last_mandatory_test_on_possible_trigger_day,

                               'last_test_on_optional_early_trigger_test_day': res.last_test_on_optional_early_trigger_test_day,
                               'last_test_on_optional_early_trigger_test_date': res.last_test_on_optional_early_trigger_test_date,
                               'last_test_on_optional_early_trigger_test_weekday': res.last_test_on_optional_early_trigger_test_weekday,
                               'prediction_on_last_optional_early_test_on_possible_trigger_day': res.prediction_on_last_optional_early_test_on_possible_trigger_day,
                               'prediction_name_on_last_optional_early_test_on_possible_trigger_day': res.prediction_name_on_last_optional_early_test_on_possible_trigger_day,

                               'last_test_on_optional_late_trigger_test_day': res.last_test_on_optional_late_trigger_test_day,
                               'last_test_on_optional_late_trigger_test_date': res.last_test_on_optional_late_trigger_test_date,
                               'last_test_on_optional_late_trigger_test_weekday': res.last_test_on_optional_late_trigger_test_weekday,
                               'prediction_on_last_optional_late_test_on_possible_trigger_day': res.prediction_on_last_optional_late_test_on_possible_trigger_day,
                               'prediction_name_on_last_optional_late_test_on_possible_trigger_day': res.prediction_name_on_last_optional_late_test_on_possible_trigger_day,

                               'true_transfer_day': res.target_transfer_day, 'true_transfer_date': res.true_transfer_date,
                               'true_transfer_weekday': res.true_transfer_weekday,
                               'predicted_ovulation_day': res.predicted_ovulation_day,
                               # 'predicted_transfer_date': res.predicted_transfer_date, 'predicted_transfer_weekday': res.predicted_transfer_weekday,
                               'transfer_day': res.predicted_transfer_day, 'transfer_date': res.predicted_transfer_date,
                               'transfer_weekday': res.predicted_transfer_weekday,
                               'transfer_weekday_name': res.predicted_transfer_weekday_name,
                               'transfer_weekday_int': res.predicted_transfer_weekday_int,
                               'trigger_day': res.trigger_day, 'trigger_date': res.trigger_date,
                               'trigger_weekday': res.trigger_weekday,
                               'num_days_ovulation_was_moved_back': res.num_days_ovulation_was_moved_back,
                               'relative_trigger_suggestion_day': res.relative_trigger_suggestion_day,
                               'relative_transfer_day': res.relative_transfer_day,

                               'unwanted_ovulation_days': res.unwanted_ovulation_days,
                               'clinic_closed_days': res.clinic_closed_days,
                               'mandatory_trigger_test_day': res.mandatory_trigger_test_day,
                               'optional_early_trigger_test_days': res.optional_early_trigger_test_days,
                               'optional_late_trigger_test_day': res.optional_late_trigger_test_day,
                               'all_possible_trigger_days': res.all_possible_trigger_days,

                               'all_previous_distance_from_true_ovulation': res.all_previous_distance_from_true_ovulation,
                               'previous_distance_from_true_ovulation': res.previous_distance_from_true_ovulation,
                               'previous_prediction': res.previous_prediction,
                               'previous_prediction_name': res.previous_prediction_name,
                               }

        all_res_df = pd.DataFrame(res_dict).T
        return all_res_df

    def calculate_treatment_results(self, log=False) -> list[OvulationPredictionTreatmentResults]:
        treatment_results = []

        if log:
            print(
                f"Calculating success rate with default initial test day: {self.treatment_manager.default_initial_test_day}")
        for ovulation_day, ovulation_day_probability in self.statistics.ovulation_day_probabilities.items():
            if ovulation_day_probability > 0:
                if log:
                    print(f"Calculating success rate for ovulation day {ovulation_day}")
                ovulation_day_treatment_results = self._calculate_treatment_results_for_cycle_start_on_all_week_days(
                    ovulation_day, ovulation_day_probability, log=False)
                treatment_results += ovulation_day_treatment_results
            else:
                if log:
                    print(f"Ovulation in day: {ovulation_day} has probability: {ovulation_day_probability}")
        return treatment_results

    def _calculate_treatment_results_for_cycle_start_on_all_week_days(self, ovulation_day: int,
                                                                      ovulation_day_probability, log: bool):
        weekday_probability = 1 / 7
        prediction_treatment_results = []

        for day in range(1, 8):
            cycle_start_date = datetime.date(self.year, self.month, day)
            unwanted_days = ModifiedFetUnwantedDays(cycle_start_date=cycle_start_date, embryo_age=self.embryo_age,
                                                    clinic_closed_dates=self.clinic_closed_dates,
                                                    clinic_minimal_staff_dates=self.clinic_minimal_staff_dates)

            adjusted_initial_test_day = self.treatment_manager.get_initial_test_day(cycle_start_date=cycle_start_date,
                                                                                    embryo_age=self.embryo_age,
                                                                                    clinic_closed_dates=self.clinic_closed_dates,
                                                                                    clinic_minimal_staff_dates=self.clinic_minimal_staff_dates)

            relative_prediction_day = adjusted_initial_test_day - ovulation_day
            weekday_ovulation_day_treatment_results = self._calculate_treatment_results(
                ovulation_day_probability * weekday_probability, (relative_prediction_day,),
                adjusted_initial_test_day,
                cycle_start_date=cycle_start_date,
                unwanted_days=unwanted_days, initial_test_day=adjusted_initial_test_day,
                test_days=[adjusted_initial_test_day],
                all_previous_predictions=[], log=log)

            prediction_treatment_results += weekday_ovulation_day_treatment_results

        return prediction_treatment_results

    def _calculate_treatment_results(self, probability: float, true_relative_prediction_days: tuple, test_day: int,
                                     cycle_start_date: datetime.date,
                                     unwanted_days: ModifiedFetUnwantedDays, initial_test_day: int, test_days: list,
                                     all_previous_predictions: list[
                                         OvulationPredictionDayPredictionWithMultipleClasses], log=False,
                                     logs_prefix="   ", has_early_warning=0):
        if true_relative_prediction_days is None:
            true_relative_prediction_days = ()
        elif isinstance(true_relative_prediction_days, (int, np.integer)):
            true_relative_prediction_days = (int(true_relative_prediction_days),)
        elif not isinstance(true_relative_prediction_days, tuple):
            true_relative_prediction_days = tuple(true_relative_prediction_days)

        latest_prediction_day = true_relative_prediction_days[-1] if true_relative_prediction_days else None
        true_ovulation_day = test_day - latest_prediction_day
        true_class = self.get_target_class(latest_prediction_day)

        if log:
            print(
                f"{logs_prefix}Calculating success rate for relative prediction days: {true_relative_prediction_days}: with true class: '{true_class}'")

        results = []

        predicted_class_probabilities = self._get_predicted_class_probabilities(true_class, test_day, initial_test_day, probability)

        for prediction_class_options, predicted_class_probability in predicted_class_probabilities.items():
            if predicted_class_probability:
                next_instruction = self.treatment_manager.get_treatment_cycle_instruction_from_prediction_class_options(
                    prediction_class_options=prediction_class_options, current_test_day=test_day,
                    embryo_age=self.embryo_age,
                    cycle_start_date=cycle_start_date, clinic_closed_dates=self.clinic_closed_dates,
                    clinic_minimal_staff_dates=self.clinic_minimal_staff_dates)
                if next_instruction.cycle_missed:
                    predicted_class_treatment_results = [
                        PossibleTreatmentResult(probability=predicted_class_probability, test_days=test_days,
                                                prediction=prediction_class_options[0],
                                                all_previous_predictions=all_previous_predictions,
                                                initial_test_day=initial_test_day, first_cycle_day=cycle_start_date,
                                                true_natural_ovulation_day=true_ovulation_day,
                                                predicted_ovulation_day_relative_to_true_ovulation=None,
                                                relative_test_day=test_day - true_ovulation_day,
                                                predicted_transfer_day=None, relative_transfer_day=None,
                                                unwanted_ovulation_days=unwanted_days.ovulation_days_that_lead_to_transfer_on_clinic_closed_days,
                                                clinic_closed_days=unwanted_days.clinic_closed_days,
                                                mandatory_trigger_test_day=unwanted_days.mandatory_trigger_test_day,
                                                optional_early_trigger_test_days=unwanted_days.optional_early_trigger_test_days,
                                                optional_late_trigger_test_day=unwanted_days.optional_late_trigger_test_day,
                                                all_possible_trigger_days=unwanted_days.all_possible_trigger_days,
                                                trigger_was_suggested=False, relative_trigger_suggestion_day=None,
                                                there_are_closed_days=self.there_are_closed_days,
                                                there_are_minimal_staff_days=self.there_are_minimal_staff_days,
                                                ovulation_was_missed=True, embryo_age=self.embryo_age, year=self.year,
                                                month=self.month)]

                else:
                    class_options_predicted_ovulation_days = self.treatment_manager.get_ovulation_predicted_cycle_day(
                        prediction_class_options[0], test_day)
                    chosen_ovulation_day_prediction = self.treatment_manager.choose_ovulation_prediction_day(
                        class_options_predicted_ovulation_days,
                        unwanted_days.ovulation_days_that_lead_to_transfer_on_clinic_closed_days)
                    predicted_ovulation_day_relative_to_true_ovulation = true_ovulation_day - chosen_ovulation_day_prediction

                    next_instruction_types = [val[0] for val in next_instruction.instructions.values()]
                    if InstructionType.OVULATION in next_instruction_types:
                        predicted_transfer_day = self.get_transfer_day(next_instruction)
                        predicted_class_treatment_results = [
                            PossibleTreatmentResult(probability=predicted_class_probability, test_days=test_days,
                                                    prediction=prediction_class_options[0],
                                                    all_previous_predictions=all_previous_predictions,
                                                    initial_test_day=initial_test_day, first_cycle_day=cycle_start_date,
                                                    true_natural_ovulation_day=true_ovulation_day,
                                                    predicted_ovulation_day_relative_to_true_ovulation=predicted_ovulation_day_relative_to_true_ovulation,
                                                    relative_test_day=test_day - true_ovulation_day,
                                                    predicted_transfer_day=predicted_transfer_day,
                                                    relative_transfer_day=predicted_transfer_day - true_ovulation_day,
                                                    unwanted_ovulation_days=unwanted_days.ovulation_days_that_lead_to_transfer_on_clinic_closed_days,
                                                    clinic_closed_days=unwanted_days.clinic_closed_days,
                                                    mandatory_trigger_test_day=unwanted_days.mandatory_trigger_test_day,
                                                    optional_early_trigger_test_days=unwanted_days.optional_early_trigger_test_days,
                                                    optional_late_trigger_test_day=unwanted_days.optional_late_trigger_test_day,
                                                    all_possible_trigger_days=unwanted_days.all_possible_trigger_days,
                                                    trigger_was_suggested=False, relative_trigger_suggestion_day=None,
                                                    there_are_closed_days=self.there_are_closed_days,
                                                    there_are_minimal_staff_days=self.there_are_minimal_staff_days,
                                                    ovulation_was_missed=False, embryo_age=self.embryo_age,
                                                    year=self.year, month=self.month)]

                    elif InstructionType.TRIGGER in next_instruction_types:
                        predicted_transfer_day = self.get_transfer_day(next_instruction)
                        trigger_day = self.get_trigger_day(next_instruction)
                        predicted_class_treatment_results = [
                            PossibleTreatmentResult(probability=predicted_class_probability, test_days=test_days,
                                                    prediction=prediction_class_options[0],
                                                    all_previous_predictions=all_previous_predictions,
                                                    initial_test_day=initial_test_day, first_cycle_day=cycle_start_date,
                                                    true_natural_ovulation_day=true_ovulation_day,
                                                    predicted_ovulation_day_relative_to_true_ovulation=predicted_ovulation_day_relative_to_true_ovulation,
                                                    relative_test_day=test_day - true_ovulation_day,
                                                    predicted_transfer_day=predicted_transfer_day,
                                                    relative_transfer_day=predicted_transfer_day - true_ovulation_day,
                                                    unwanted_ovulation_days=unwanted_days.ovulation_days_that_lead_to_transfer_on_clinic_closed_days,
                                                    clinic_closed_days=unwanted_days.clinic_closed_days,
                                                    mandatory_trigger_test_day=unwanted_days.mandatory_trigger_test_day,
                                                    optional_early_trigger_test_days=unwanted_days.optional_early_trigger_test_days,
                                                    optional_late_trigger_test_day=unwanted_days.optional_late_trigger_test_day,
                                                    all_possible_trigger_days=unwanted_days.all_possible_trigger_days,
                                                    trigger_was_suggested=True,
                                                    relative_trigger_suggestion_day=trigger_day - true_ovulation_day,
                                                    there_are_closed_days=self.there_are_closed_days,
                                                    there_are_minimal_staff_days=self.there_are_minimal_staff_days,
                                                    ovulation_was_missed=False, embryo_age=self.embryo_age,
                                                    year=self.year, month=self.month)]

                    elif InstructionType.BLOOD_TEST in next_instruction_types:
                        if isinstance(prediction_class_options, tuple):
                            if len(prediction_class_options) == 1:
                                if prediction_class_options[0] == self.model_target_classes_in_order[0]:
                                    next_test_will_have_early_warning = 0
                                else:
                                    next_test_will_have_early_warning = 1
                            else:
                                next_test_will_have_early_warning = 1
                        else:
                            if prediction_class_options == self.model_target_classes_in_order[0]:
                                next_test_will_have_early_warning = 0
                            else:
                                next_test_will_have_early_warning = 1

                        next_test_day = [day for day, instruction in next_instruction.instructions.items() if
                                         InstructionType.BLOOD_TEST in instruction][0]

                        updated_test_days = test_days + [next_test_day]
                        updated_previous_predictions = all_previous_predictions + [prediction_class_options[0]]

                        next_relative_prediction_day = latest_prediction_day + (next_test_day - test_day)
                        next_prediction_days = true_relative_prediction_days + (next_relative_prediction_day,)
                        predicted_class_treatment_results = self._calculate_treatment_results(
                            predicted_class_probability, next_prediction_days, next_test_day,
                            cycle_start_date, unwanted_days, initial_test_day=initial_test_day,
                            test_days=updated_test_days,
                            all_previous_predictions=updated_previous_predictions, log=log,
                            logs_prefix=logs_prefix + "   ",
                            has_early_warning=next_test_will_have_early_warning)

                results += predicted_class_treatment_results
            else:
                if log:
                    print(
                        f"{logs_prefix}Skipping prediction class options '{prediction_class_options}' because the probability to predict it from true class '{true_class}' is 0")
        return results

    def _generate_target_classes(self) -> None:
        # Fixed range for model classes -6 to +2 based on standard model outputs.
        self.target_classes = [-6, -5, -4, -3, -2, -1, 0, 1, 2]

    def _get_predicted_class_probabilities(self,
                                           true_class: str,
                                           test_day: int,
                                           initial_test_day: int,
                                           current_probability: float,
                                           minimum_probability_to_consider: float = 0.00000000001) -> dict[
        tuple[OvulationPredictionDayPredictionWithMultipleClasses], float]:

        relative_day_enum = RelativeOvulationDay.get_relative_day_from_str_to_enum(true_class)
        predicted_class_probabilities_by_initial_test_day = self.predicted_class_probabilities[initial_test_day]

        if test_day <= initial_test_day + 2:
            predictions_confusion_matrix = predicted_class_probabilities_by_initial_test_day[
                self.ONE_TEST_BEFORE_OR_EQUAL_DAY_10_CM]
            predicted_class_probabilities = predictions_confusion_matrix[(relative_day_enum,)]
        else:
            predictions_confusion_matrix = predicted_class_probabilities_by_initial_test_day[self.TWO_TESTS_CM]
            predicted_class_probabilities = predictions_confusion_matrix[(relative_day_enum,)]

        predicted_class_probabilities_after_filters = {
            k: v for k, v in predicted_class_probabilities.items()
            if v * current_probability > minimum_probability_to_consider
        }

        if not predicted_class_probabilities_after_filters:
            predicted_class_probabilities_after_filters = {k: v for k, v in predicted_class_probabilities.items() if v > 0}

        if not predicted_class_probabilities_after_filters:
            return {}

        total = sum(predicted_class_probabilities_after_filters.values())
        if not total:
            return {}

        normalized_predicted_class_probabilities = {k: v / total for k, v in predicted_class_probabilities_after_filters.items()}
        normalized_with_current_probability_predicted_class_probabilities = {k: v * current_probability for k, v in normalized_predicted_class_probabilities.items()}

        return normalized_with_current_probability_predicted_class_probabilities

