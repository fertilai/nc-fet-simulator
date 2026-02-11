from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.interfaces.treatment_management import InstructionType

from ovufet_sim.io import (
    load_precomputed_cm,
    load_ovulation_day_probabilities,
)
from ovufet_sim.simulator import (
    OvulationPredictionTreatmentResultsCalculator,
)
from src.treatments.fet.selective_modified_natural import SelectiveModifiedNaturalFetTreatmentManager


DEFAULT_INITIAL_TEST_DAY = 8
MIN_RELATIVE_PREDICTION_MARGIN = 0.3


# Some upstream slices don't include OVULATION on InstructionType, but the simulator expects it as a sentinel.
if not hasattr(InstructionType, "OVULATION"):
    InstructionType.OVULATION = "OVULATION"


def export_representative_confusion_matrix(cm: dict, out_path: Path) -> None:
    initial_day = DEFAULT_INITIAL_TEST_DAY
    bucket = OvulationPredictionTreatmentResultsCalculator.TWO_TESTS_CM
    cm_dict = cm[initial_day][bucket]

    rows = []
    for true_k, pred_map in cm_dict.items():
        true_lbl = str(true_k[0])
        for pred_k, p in pred_map.items():
            pred_lbl = str(pred_k[0])
            rows.append((true_lbl, pred_lbl, float(p)))

    df = pd.DataFrame(rows, columns=["true_class", "pred_class", "probability"])
    pivot = df.pivot(index="true_class", columns="pred_class", values="probability").fillna(0.0)
    pivot.to_csv(out_path, index=True)


def summarize_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    w = results_df["probability"].astype(float)
    wsum = float(w.sum()) if float(w.sum()) > 0 else 1.0

    succ = results_df["success"].astype(bool)
    missed = results_df["ovulation_was_missed"].astype(bool)
    incorrect = (~succ) & (~missed)

    correct_rate = float(w[succ].sum()) / wsum
    no_pred_rate = float(w[missed].sum()) / wsum
    incorrect_rate = float(w[incorrect].sum()) / wsum
    avg_tests = float((w * results_df["num_tests"].astype(float)).sum()) / wsum

    return pd.DataFrame([{
        "Correct Prediction Rate (Success)": correct_rate,
        "Incorrect Prediction Rate (Failure)": incorrect_rate,
        "No Prediction Rate (Cancellation)": no_pred_rate,
        "Average Number of Tests": avg_tests,
        "Total Probability Mass": wsum,
        "N Outcomes (rows)": int(len(results_df)),
    }])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ovufet-sim")
    ap.add_argument("--precomputed-cm", required=True, help="Path to precomputed CM dict text file.")
    ap.add_argument("--ovulation-day-probs", required=True, help="Path to ovulation day probabilities JSON file.")
    ap.add_argument("--out-dir", default="artifacts", help="Directory to write outputs.")
    ap.add_argument("--export-results", action="store_true", help="Write treatment_results.csv (can be large).")
    args = ap.parse_args(argv)

    cm_path = Path(args.precomputed_cm).expanduser().resolve()
    probs_path = Path(args.ovulation_day_probs).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = load_precomputed_cm(cm_path)
    ovulation_day_probabilities = load_ovulation_day_probabilities(probs_path)

    model = None  # This journal slice uses CM + eval Excel and does not require the trained model object.
    tm = SelectiveModifiedNaturalFetTreatmentManager(
        ovulation_prediction_model=model,
        default_initial_test_day=DEFAULT_INITIAL_TEST_DAY,
        min_relative_prediction_margin=MIN_RELATIVE_PREDICTION_MARGIN,
    )

    calc = OvulationPredictionTreatmentResultsCalculator(
        model=model,
        first_test_day_name="first_day",
        second_test_day_name="second_day",
        treatment_cycles=[],
        treatment_manager=tm,
        randomize_confusion_matrices=False,
        print_cm=False,
        predicted_class_probabilities=cm,
        ovulation_day_probabilities=ovulation_day_probabilities,
    )

    results = calc.calculate_treatment_results()

    # Summary without materializing a dataframe (fast, low-memory).
    wsum = 0.0
    w_succ = 0.0
    w_missed = 0.0
    w_tests = 0.0
    n = 0

    for r in results:
        p = float(getattr(r, "probability", 0.0) or 0.0)
        wsum += p
        n += 1
        if bool(getattr(r, "success", False)):
            w_succ += p
        if bool(getattr(r, "ovulation_was_missed", False)):
            w_missed += p
        w_tests += p * float(getattr(r, "num_tests", 0) or 0)

    if wsum <= 0:
        wsum = 1.0

    summary = pd.DataFrame([{
        "Correct Prediction Rate (Success)": w_succ / wsum,
        "Incorrect Prediction Rate (Failure)": (wsum - w_succ - w_missed) / wsum,
        "No Prediction Rate (Cancellation)": w_missed / wsum,
        "Average Number of Tests": w_tests / wsum,
        "Total Probability Mass": wsum,
        "N Outcomes (rows)": n,
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)

    export_representative_confusion_matrix(cm, out_dir / "confusion_matrix.csv")

    print(summary)


if __name__ == "__main__":
    main()
