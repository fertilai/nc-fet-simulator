from enum import Enum

class Weekday(Enum):
    MONDAY=0
    TUESDAY=1
    WEDNESDAY=2
    THURSDAY=3
    FRIDAY=4
    SATURDAY=5
    SUNDAY=6

class InstructionType(str, Enum):
    MEDICATION="MEDICATION"
    ULTRASOUND="ULTRASOUND"
    BLOOD_TEST="BLOOD_TEST"
    TRANSFER="TRANSFER"
    CANCEL="CANCEL"
    OTHER="OTHER"
    OVULATION="OVULATION" # Added because it was patched in cli.py/simulator.py
    TRIGGER="TRIGGER"     # Added because it was patched in simulator.py
    DECLARE_CYCLE_MISSED="DECLARE_CYCLE_MISSED" # Added because it was patched in simulator.py

class HormoneName(str, Enum):
    LH="LH"
    E2="E2"
    P4="P4"
    HCG="HCG"

class UltrasoundMeasurementName(str, Enum):
    ENDOMETRIUM="ENDOMETRIUM"
    FOLLICLE="FOLLICLE"

class IssueSeverity(str, Enum):
    INFO="INFO"
    WARNING="WARNING"
    ERROR="ERROR"

class OvulationPredictionDayTargetClass(str, Enum):
    MinusSixOrLower = "MinusSixOrLower"
    MinusFive = "MinusFive"
    MinusFour = "MinusFour"
    MinusThree = "MinusThree"
    MinusTwo = "MinusTwo"
    MinusOne = "MinusOne"
    Zero = "Zero"
    PlusOne = "PlusOne"
    PlusTwoOrHigher = "PlusTwoOrHigher"

# Attached property for OvulationPredictionDayTargetClass
OvulationPredictionDayTargetClass.convert_num_label_to_str = {
    "MinusSixOrLower": "-6",
    "MinusFive": "-5",
    "MinusFour": "-4",
    "MinusThree": "-3",
    "MinusTwo": "-2",
    "MinusOne": "-1",
    "Zero": "0",
    "PlusOne": "+1",
    "PlusTwoOrHigher": "+2",
}

class OvulationPredictionDayPredictionWithMultipleClasses(str, Enum):
    MinusSixOrLower = "MinusSixOrLower"
    MinusSixOrMinusFive = "MinusSixOrMinusFive"
    MinusFive = "MinusFive"
    MinusFiveOrMinusFour = "MinusFiveOrMinusFour"
    MinusFour = "MinusFour"
    MinusFourOrMinusThree = "MinusFourOrMinusThree"
    MinusThree = "MinusThree"
    MinusThreeOrMinusTwo = "MinusThreeOrMinusTwo"
    MinusTwo = "MinusTwo"
    MinusTwoOrMinusOne = "MinusTwoOrMinusOne"
    MinusOne = "MinusOne"
    MinusOneOrZero = "MinusOneOrZero"
    Zero = "Zero"
    ZeroOrPlusOne = "ZeroOrPlusOne"
    PlusOne = "PlusOne"
    PlusTwoOrHigher = "PlusTwoOrHigher"

class CompareType(str, Enum):
    LT='LT'
    LTE='LTE'
    EQ='EQ'
    GTE='GTE'
    GT='GT'

class IssueActionRequired(str, Enum):
    DONT_DISPLAY_ALGORITHM_RESULTS="DONT_DISPLAY_ALGORITHM_RESULTS"
