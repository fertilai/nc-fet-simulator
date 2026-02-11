# Ovulation FET simulator

This repository is a **minimal, deterministic** simulator used to reproduce the paper KPIs from:

1. An evaluation Excel file (model outputs per test), and  
2. A **precomputed confusion-matrix dictionary** exported from the original codebase.

The simulator runs **probability-weighted recursion**, so results are reproducible.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --index-url https://pypi.org/simple
```

# NC-FET Simulator

A minimal, deterministic simulator for Natural Cycle Frozen Embryo Transfer (NC-FET) treatments.
This repository contains the simulation engine (`ovufet_sim`) and the core interfaces.

The simulation logic depends on an implementation of the `FetTreatmentManager` interface, which defines clinical decision logic.

## Inputs & Outputs

### Inputs

To run the simulator, you must provide:
1.  **Precomputed Confusion Matrix** (`precomputed_cm.txt`): A dictionary defining the probabilistic performance of the ovulation prediction model.
2.  **Ovulation Day Probabilities** (`ovulation_day_probabilities.json`): A distribution of natural ovulation days used to weight the simulation scenarios.

### Outputs

The simulator produces the following artifacts in the `--out-dir` (default: `artifacts/`):
*   **`summary.csv`**: Aggregated Key Performance Indicators (KPIs).
    *   `Correct Prediction Rate (Success)`: Probability that transfer occurs on the optimal day.
    *   `Incorrect Prediction Rate (Failure)`: Probability of suboptimal timing (error).
    *   `No Prediction Rate (Cancellation)`: Probability of cycle cancellation due to missed ovulation.
    *   `Average Number of Tests`: Weighted average of blood tests performed.
*   **`confusion_matrix.csv`**: A representative slice of the probability transitions (typically for day 8, two-tests bucket).

## Architecture

The project is structured to separate the generic simulation engine from the specific clinical protocol.

*   `ovufet_sim/simulator.py`: Contains the `OvulationPredictionTreatmentResultsCalculator`, which recursively calculates outcomes based on probabilities. It relies on the `FetTreatmentManager` interface.
*   `src/interfaces/treatment_management.py`: Defines the `FetTreatmentManager` abstract base class.
*   **Implementation**: Users must provide a concrete implementation of `FetTreatmentManager` (e.g., `SelectiveModifiedNaturalFetTreatmentManager`) to define how test results translate into clinical instructions (tests, trigger, transfer).

## Run

To run the simulator with your specific implementation linked in `cli.py`:

```bash
python -m ovufet_sim.cli \
  --precomputed-cm "artifacts/precomputed_cm.txt" \
  --ovulation-day-probs "artifacts/ovulation_day_probabilities.json" \
  --out-dir "artifacts"
```


This also writes `artifacts/treatment_results.csv`.

## Input format assumptions (eval Excel)

The CLI auto-selects the best sheet and expects these columns:
- `ovulation_day_target`
- `ovulation_prediction_result`
- `ovulation_prediction_probability`
- `first_day`, `second_day`
- plus any number of `prediction probability <int>` columns used by the original export.