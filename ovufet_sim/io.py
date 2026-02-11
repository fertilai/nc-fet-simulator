from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from src.algorithms.ovulation_prediction import (
    OvulationPredictionDayPredictionWithMultipleClasses,
    RelativeOvulationDay,
)



def load_precomputed_cm(path: Path) -> dict:
    """Load a CM dict exported from the original codebase.

    The file is expected to be a Python-literal dict (not JSON) that contains keys such as:
      (<OvulationPredictionDayPredictionWithMultipleClasses.MinusSixOrLower: 0>,)

    We rewrite enum reprs into strings, parse with ast.literal_eval, then convert back to enums.
    """
    txt = Path(path).expanduser().read_text(encoding="utf-8", errors="ignore").strip()

    txt = re.sub(
        r"<OvulationPredictionDayPredictionWithMultipleClasses\.([A-Za-z0-9_]+):\s*[^>]+>",
        r"'\1'",
        txt,
    )

    raw = ast.literal_eval(txt)

    enum_to_int = RelativeOvulationDay.convert_enum_label_to_int
    int_to_enum = {v: k for k, v in enum_to_int.items() if v is not None}

    def to_pred_enum(x):
        if isinstance(x, OvulationPredictionDayPredictionWithMultipleClasses):
            return x
        if isinstance(x, str):
            if x in OvulationPredictionDayPredictionWithMultipleClasses.__members__:
                return OvulationPredictionDayPredictionWithMultipleClasses[x]
            try:
                return OvulationPredictionDayPredictionWithMultipleClasses(x)
            except Exception:
                pass
            try:
                xi = int(x)
                if xi in int_to_enum:
                    return int_to_enum[xi]
            except Exception:
                pass
        if isinstance(x, (int, float)) and int(x) in int_to_enum:
            return int_to_enum[int(x)]
        raise ValueError(f"Cannot map CM key to enum: {x!r}")

    def convert(obj, level=0):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if level == 0:
                    nk = int(k)
                elif level == 1:
                    nk = str(k)
                else:
                    if isinstance(k, tuple):
                        if len(k) != 1:
                            raise ValueError(f"Unexpected tuple key length: {k!r}")
                        nk = (to_pred_enum(k[0]),)
                    else:
                        nk = (to_pred_enum(k),)
                out[nk] = convert(v, level + 1)
            return out
        if isinstance(obj, list):
            return [convert(x, level) for x in obj]
        if isinstance(obj, tuple):
            return tuple(convert(x, level) for x in obj)
        return obj

    cm = convert(raw)

    any_initial = next(iter(cm.keys()))
    any_bucket = next(iter(cm[any_initial].keys()))
    any_true = next(iter(cm[any_initial][any_bucket].keys()))
    if not isinstance(any_true, tuple) or not isinstance(any_true[0], OvulationPredictionDayPredictionWithMultipleClasses):
        raise RuntimeError("Loaded CM does not have tuple(enum) keys as expected")

    return cm


def load_ovulation_day_probabilities(path: Path) -> dict[int, float]:
    with open(Path(path).expanduser().resolve()) as f:
        data = json.load(f)
    # Convert keys to int (json keys are strings)
    return {int(k): float(v) for k, v in data.items()}
