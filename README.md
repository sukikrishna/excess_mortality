# Excess Mortality Forecasting: Statistical vs. Deep Learning Models

This repository contains the full experimental pipeline for the paper:

> **Statistical vs. Deep Learning Models for Estimating Substance Overdose Excess Mortality in the US**
> arXiv: [2512.21456](https://arxiv.org/abs/2512.21456)

We compare six time series forecasting models — SARIMA, LSTM, TCN, Seq2Seq, Seq2Seq with Attention, and Transformer — for predicting monthly drug overdose deaths in the United States across variable forecast horizons (1–4 years). Models are trained on pre-pandemic data (2015–2019) and evaluated on the COVID-era period (2020–2023) to quantify excess mortality.

---

## Table of Contents

- [Data](#data)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Experiment Pipeline](#experiment-pipeline)
  - [Step 1: Sensitivity Analysis (Optional)](#step-1-sensitivity-analysis-optional)
  - [Step 2: Hyperparameter Grid Search](#step-2-hyperparameter-grid-search)
  - [Step 3: Final Evaluation](#step-3-final-evaluation)
  - [Step 4: Plotting & Metrics](#step-4-plotting--metrics)
- [Script Reference](#script-reference)
- [Results](#results)
- [Citation](#citation)

---

## Data

Data comes from the **CDC WONDER** (Wide-ranging Online Data for Epidemiologic Research) database, available at [wonder.cdc.gov](https://wonder.cdc.gov/).

### How to obtain the data

1. Go to [CDC WONDER](https://wonder.cdc.gov/) and select **Multiple Cause of Death (Provisional)**.
2. Under **Group Results By**, select: `State`, `Month`.
3. Under **Select Causes of Death**, filter to drug/substance overdose ICD-10 codes (e.g., T36–T50 for poisoning by drugs, medicaments, and biological substances).
4. Set the date range to cover your analysis window (e.g., January 2015 – December 2023).
5. Export the result as a tab-delimited text file and convert to `.xlsx`.

The data files used in this paper are included in the `data/` directory:

| File | Description |
|------|-------------|
| `data/state_month_overdose.xlsx` | State-level monthly overdose deaths (original dataset used for grid search) |
| `data/state_month_overdose_2015_2023.xlsx` | State-level monthly overdose deaths, 2015–2023 (used for sensitivity analysis and variable-horizon evaluation) |
| `data/national_month_overdose_2010_2023.xlsx` | National-level monthly overdose deaths, 2010–2023 (used for national-level final evaluation) |

> **Note:** Suppressed counts (cells where CDC suppresses data for privacy when counts are < 10) are treated as 0. Data is aggregated to monthly national totals prior to modeling.

---

## Installation

Python 3.8+ is recommended. Install dependencies manually:

```bash
pip install tensorflow statsmodels scikit-learn pandas numpy matplotlib seaborn scipy keras-tcn openpyxl
```

Key packages:

| Package | Purpose |
|---------|---------|
| `tensorflow` | LSTM, Seq2Seq, Transformer models |
| `keras-tcn` | Temporal Convolutional Network (TCN) |
| `statsmodels` | SARIMA model |
| `scikit-learn` | Preprocessing and metrics |
| `scipy` | Statistical tests, confidence intervals |
| `openpyxl` | Reading Excel data files |

---

## Repository Structure

```
excess_mortality/
│
├── data/                                    # Input data files (CDC WONDER exports)
│
├── experiments-up.py                        # Hyperparameter grid search (all models)
├── run_all_grid_search.py                   # Orchestrator for running grid search across all models
│
├── fast_sensitivity-random.py               # Fast sensitivity analysis (LSTM proxy, random seeds)
├── fast_sensitivity.py                      # Fast sensitivity analysis (LSTM proxy, fixed seeds)
├── sensitivity.py                           # Full sensitivity analysis (all model types)
├── sensitivity_log.py                       # Sensitivity analysis with checkpoint/resume support
│
├── hyperparameters_static.py                # Extracts and documents optimal hyperparameters from grid search
│
├── final_evaluation.py                      # Final model evaluation (state-level and national-level)
├── run_evaluation.py                        # Orchestrator script for running full evaluation pipeline
│
├── metrics.py                               # Variable-horizon evaluation (separate train per horizon)
├── metrics-singletrain.py                   # Variable-horizon evaluation (single train, extract horizons)
├── metrics-final.py                         # Plotting and CSV export of variable-horizon results
├── fix-plots-with-pi.py                     # Regenerate comparison plots with prediction intervals
│
├── efficient_evaluation_results_more_trials/  # Saved results: variable-horizon (single-train approach)
├── final_evaluation_results/                  # Saved results: state-level final evaluation
├── national_final_evaluation_results/         # Saved results: national-level final evaluation
├── og_final_evaluation_results/               # Saved results: earlier version of final evaluation
├── enhanced_model_comparison_plots_fixed/     # Output comparison plots
└── horizon_plotting_data_csv/                 # CSV exports of horizon-level plotting data
```

---

## Experiment Pipeline

The experiments in the paper follow four stages. Run them in order for full replication.

---

### Step 1: Sensitivity Analysis (Optional)

Before running a large grid search, we conducted a sensitivity analysis to determine the minimum number of trials and random seeds needed for stable model rankings. This step is optional if you want to skip directly to grid search.

**Fast version (recommended — LSTM proxy, ~minutes):**

```bash
# Using random seed generation (recommended, avoids seed correlations)
python fast_sensitivity-random.py --max_trials 50 --max_seeds 50 --eval_interval 5

# Using fixed seed generation
python fast_sensitivity.py --max_trials 50 --max_seeds 50 --eval_interval 5
```

**Full version (all model types, ~hours):**

```bash
# With progress saving and resume capability
python sensitivity_log.py --max_trials 100 --max_seeds 100 --eval_interval 5

# Resume after interruption
python sensitivity_log.py --resume
```

**What this produces:**
- Heatmaps of RMSE stability vs. (trials × seeds) combinations
- Coefficient of variation analysis
- A recommended (trials, seeds) configuration for the grid search
- Outputs saved to `sensitivity_analysis/` or `fast_sensitivity_analysis_3/`

**Key finding from the paper:** 30 trials per configuration (with a fixed seed) provides stable rankings with a coefficient of variation below 0.05 for LSTM-based models.

---

### Step 2: Hyperparameter Grid Search

The grid search trains every model across all hyperparameter combinations and saves metrics for train and validation splits.

**Run all models:**

```bash
python run_all_grid_search.py
```

**Run specific models:**

```bash
python run_all_grid_search.py --models lstm,sarima,transformer
```

**Resume an interrupted grid search:**

```bash
python run_all_grid_search.py --resume
```

**Only analyze existing results (no retraining):**

```bash
python run_all_grid_search.py --analyze-only
```

Internally, `run_all_grid_search.py` calls `experiments-up.py` for each model type. You can also run `experiments-up.py` directly by setting `MODEL_TYPE` at the top of the file:

```python
MODEL_TYPE = 'lstm'   # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer'
TRIAL_MODE = 'fixed_seed'  # 'fixed_seed' or 'multi_seed'
```

**Hyperparameter search space:**

| Parameter | Values |
|-----------|--------|
| Lookback window | 3, 5, 7, 9, 11, 12 |
| Batch size | 8, 16, 32 |
| Epochs | 50, 100 |
| Encoder/Decoder units (Seq2Seq) | 64, 128 |
| d_model / heads (Transformer) | 64 / 2 |

Grid search results are saved to `results/<model_type>/`.

**Extract optimal hyperparameters:**

After the grid search, run the following to parse and display the best configuration per model:

```bash
python hyperparameters_static.py
```

This outputs `optimal_hyperparameters_config.py` with the best hyperparameter set for each model based on validation RMSE.

**Optimal hyperparameters used in the paper:**

| Model | Lookback | Batch Size | Epochs | Other |
|-------|----------|------------|--------|-------|
| LSTM | 5 | 8 | 50 | — |
| TCN | 5 | 32 | 50 | — |
| Seq2Seq | 7 | 16 | 100 | enc=64, dec=64 |
| Seq2Seq+Attn | 5 | 16 | 50 | enc=128, dec=64 |
| Transformer | 7 | 32 | 100 | d_model=64, heads=2 |
| SARIMA | — | — | — | order=(2,1,1), seasonal=(1,1,1,12) |

---

### Step 3: Final Evaluation

After selecting optimal hyperparameters, train each model on the full train+validation set (2015–2019) and evaluate on the test set (2020 onward). Multiple seeds and trials are used for robust statistics.

**State-level evaluation:**

```bash
python final_evaluation.py
```

By default this runs 5 seeds × 10 trials per model and saves to `final_evaluation_results/`. To switch to the national-level dataset, change `DATA_PATH` at the top of the file:

```python
DATA_PATH = 'data/national_month_overdose_2010_2023.xlsx'
RESULTS_DIR = 'national_final_evaluation_results'
```

**Variable-horizon evaluation (recommended — efficient):**

Trains each model once and extracts predictions at 12, 24, 36, and 48 month horizons from the same run:

```bash
python metrics-singletrain.py
```

This is more efficient than `metrics.py` (which retrains for each horizon) and is the approach used for the paper's horizon analysis. Results are saved to `efficient_evaluation_results_more_trials/`.

**Alternative (separate train per horizon):**

```bash
python metrics.py
```

Saves results to `comprehensive_evaluation_results/`.

---

### Step 4: Plotting & Metrics

**Generate comparison plots and CSV exports from variable-horizon results:**

```bash
python metrics-final.py
```

This reads from `efficient_evaluation_results_more_trials/` and produces:
- Per-horizon comparison plots (SARIMA vs. each deep learning model)
- 4-panel overview: RMSE vs. horizon, MAPE vs. horizon, model rankings heatmap, performance degradation
- CSV files with raw plotting data in `horizon_plotting_data_csv/`

**Regenerate plots with prediction intervals:**

```bash
python fix-plots-with-pi.py
```

Outputs enhanced plots to `enhanced_model_comparison_plots_fixed/`.

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `experiments-up.py` | Core grid search loop. Trains and evaluates each model across hyperparameter combinations for a given `MODEL_TYPE`. Saves per-trial CSVs and summary metrics. |
| `run_all_grid_search.py` | Orchestrator that runs `experiments-up.py` for each model type sequentially. Supports resume, model filtering, and post-hoc analysis. |
| `fast_sensitivity-random.py` | Runs sensitivity analysis using LSTM as a proxy to determine stable (trials, seeds) combinations. Uses properly randomized seeds to avoid correlations. |
| `fast_sensitivity.py` | Same as above but with arithmetic seed generation (earlier version). |
| `sensitivity.py` | Full sensitivity analysis across all 10 model configurations (LSTM, TCN, SARIMA, Seq2Seq, Transformer variants). |
| `sensitivity_log.py` | Full sensitivity analysis with checkpoint saving, graceful interruption handling, and resume support. Preferred for long runs. |
| `hyperparameters_static.py` | Parses grid search CSVs to extract the best hyperparameter configuration per model based on validation RMSE. |
| `final_evaluation.py` | Trains each model with optimal hyperparameters on train+val data, evaluates on test set. Computes RMSE, MAE, MAPE, and 95% prediction interval coverage across multiple seeds/trials. |
| `run_evaluation.py` | Orchestrator for the full final evaluation pipeline (evaluation → extraction → plotting). |
| `metrics.py` | Variable-horizon evaluation that retrains separately for each horizon (2020, 2020–2021, 2020–2022, 2020–2023). More computationally expensive. |
| `metrics-singletrain.py` | Efficient variable-horizon evaluation: trains once per model on 2015–2019 data, forecasts the full 2020–2023 sequence, then slices predictions to each horizon. Preferred approach. |
| `metrics-final.py` | Loads results from `efficient_evaluation_results_more_trials/` and generates all comparison plots and CSV exports used in the paper. |
| `fix-plots-with-pi.py` | Regenerates per-model comparison plots with confidence intervals and prediction intervals. |

---

## Results

Pre-computed results are included in the repository:

| Directory | Contents |
|-----------|---------|
| `efficient_evaluation_results_more_trials/` | Variable-horizon metrics (50 trials per model), predictions, and plots — primary results used in the paper |
| `final_evaluation_results/` | State-level final evaluation across 5 seeds × 10 trials |
| `national_final_evaluation_results/` | National-level final evaluation |
| `og_final_evaluation_results/` | Earlier version of final evaluation results |
| `enhanced_model_comparison_plots_fixed/` | Publication-quality comparison plots |
| `horizon_plotting_data_csv/` | Raw CSV exports of all plotted data |

To reproduce results from scratch, follow the pipeline in order: sensitivity analysis → grid search → final evaluation → plotting.

---

## Citation

If you use this code or data in your work, please cite:

```bibtex
@article{excess_mortality_2024,
  title={Statistical vs. Deep Learning Models for Estimating Substance Overdose Excess Mortality in the US},
  author={Krishna, Sukirth and others},
  journal={arXiv preprint arXiv:2512.21456},
  year={2024}
}
```

---

## Notes on Reproducibility

- All deep learning models use `numpy` and `tensorflow` random seeds, but exact numerical results may vary across hardware and TensorFlow versions due to GPU non-determinism.
- SARIMA results are fully deterministic given a fixed dataset and `disp=False`.
- The sensitivity analysis scripts document the number of trials and seeds required for stable rankings. The paper uses 30 trials per configuration across 5 seeds.
- `Suppressed` values in CDC WONDER exports (counts < 10) are imputed as 0 throughout.
