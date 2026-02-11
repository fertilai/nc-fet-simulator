from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from src.interfaces.enums import (
    OvulationPredictionDayTargetClass,
    OvulationPredictionDayPredictionWithMultipleClasses,
)


class RelativeOvulationDay:
    covert_sign_str_to_int: Dict[str, int] = {"+": 1, "-": -1}
    convert_str_number_to_int: Dict[str, int] = {str(i): i for i in range(0, 10)}
    convert_int_to_str: Dict[int, str] = {
        -6: "-6",
        -5: "-5",
        -4: "-4",
        -3: "-3",
        -2: "-2",
        -1: "-1",
        0: "0",
        1: "+1",
        2: "+2",
    }
    convert_enum_label_to_int: Dict[str, int] = {
        "MinusSixOrLower": -6,
        "MinusFive": -5,
        "MinusFour": -4,
        "MinusThree": -3,
        "MinusTwo": -2,
        "MinusOne": -1,
        "Zero": 0,
        "PlusOne": 1,
        "PlusTwoOrHigher": 2,
        "MinusSixOrMinusFive": -6,
        "MinusFiveOrMinusFour": -5,
        "MinusFourOrMinusThree": -4,
        "MinusThreeOrMinusTwo": -3,
        "MinusTwoOrMinusOne": -2,
        "MinusOneOrZero": -1,
        "ZeroOrPlusOne": 0,
    }

    @classmethod
    def get_relative_day_from_enum_to_int(cls, v: Enum) -> int:
        return cls.convert_enum_label_to_int[str(v).split(".")[-1]]

    @classmethod
    def get_relative_day_from_enum_to_str(cls, v: Enum) -> str:
        return cls.get_relative_day_from_int_to_str(cls.get_relative_day_from_enum_to_int(v))

    @classmethod
    def get_relative_day_from_int_to_str(cls, v: int) -> str:
        if v <= -6:
            return "-6"
        if v >= 2:
            return "+2"
        return cls.convert_int_to_str[int(v)]

    @classmethod
    def get_relative_day_from_int_to_enum(cls, v: int) -> OvulationPredictionDayTargetClass:
        if v <= -6:
            return OvulationPredictionDayTargetClass.MinusSixOrLower
        if v >= 2:
            return OvulationPredictionDayTargetClass.PlusTwoOrHigher
        mapping = {
            -5: OvulationPredictionDayTargetClass.MinusFive,
            -4: OvulationPredictionDayTargetClass.MinusFour,
            -3: OvulationPredictionDayTargetClass.MinusThree,
            -2: OvulationPredictionDayTargetClass.MinusTwo,
            -1: OvulationPredictionDayTargetClass.MinusOne,
            0: OvulationPredictionDayTargetClass.Zero,
            1: OvulationPredictionDayTargetClass.PlusOne,
        }
        return mapping[int(v)]

    @classmethod
    def get_relative_day_from_str_to_enum(cls, s: str) -> OvulationPredictionDayTargetClass:
        s = str(s).strip()
        if s.startswith("+"):
            n = int(s[1:])
        else:
            n = int(s)
        return cls.get_relative_day_from_int_to_enum(n)

    @classmethod
    def get_relative_day_from_relative_day_to_enum(cls, rel: "RelativeOvulationDay") -> OvulationPredictionDayTargetClass:
        return cls.get_relative_day_from_int_to_enum(int(rel))

    @classmethod
    def get_relative_day_from_original_prediction_result_to_int(cls, s: str) -> int:
        s = str(s).strip()
        if s in OvulationPredictionDayTargetClass._member_map_:
            return cls.convert_enum_label_to_int[s]
        if s in OvulationPredictionDayPredictionWithMultipleClasses._member_map_:
            return cls.convert_enum_label_to_int[s]
        if s.startswith("+"):
            return int(s[1:])
        return int(s)

    @classmethod
    def convert_target_class_to_predictions_with_margins(
        cls, target: OvulationPredictionDayTargetClass, include_adjacent: bool = True
    ) -> List[OvulationPredictionDayPredictionWithMultipleClasses]:
        t = target.value if isinstance(target, OvulationPredictionDayTargetClass) else str(target)
        if not include_adjacent:
            if t in OvulationPredictionDayPredictionWithMultipleClasses._member_map_:
                return [OvulationPredictionDayPredictionWithMultipleClasses[t]]
            return [OvulationPredictionDayPredictionWithMultipleClasses[t]]
        # Adjacent-pair labels used by notebook logic
        adj = {
            "MinusSixOrLower": ["MinusSixOrLower", "MinusSixOrMinusFive"],
            "MinusFive": ["MinusFive", "MinusSixOrMinusFive", "MinusFiveOrMinusFour"],
            "MinusFour": ["MinusFour", "MinusFiveOrMinusFour", "MinusFourOrMinusThree"],
            "MinusThree": ["MinusThree", "MinusFourOrMinusThree", "MinusThreeOrMinusTwo"],
            "MinusTwo": ["MinusTwo", "MinusThreeOrMinusTwo", "MinusTwoOrMinusOne"],
            "MinusOne": ["MinusOne", "MinusTwoOrMinusOne", "MinusOneOrZero"],
            "Zero": ["Zero", "MinusOneOrZero", "ZeroOrPlusOne"],
            "PlusOne": ["PlusOne", "ZeroOrPlusOne"],
            "PlusTwoOrHigher": ["PlusTwoOrHigher"],
        }
        if t not in adj:
            # If an unexpected enum member/value slips through, fall back to the closest valid target.
            if isinstance(target, OvulationPredictionDayTargetClass):
                t = target.value
            else:
                t = str(target)
            if t not in adj:
                # Last resort: default to Zero
                t = "Zero"
        return [OvulationPredictionDayPredictionWithMultipleClasses[a] for a in adj[t]]

