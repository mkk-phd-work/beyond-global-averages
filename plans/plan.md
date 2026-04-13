Publication-Quality Implementation Plan
Objective
Update the M5 exogenous notebook workflow to:

remove manual preprocessing standardization
use internal forecaster transformation for scaling
add Optuna-based lag and hyperparameter tuning
produce publication-quality outputs for Elsevier Procedia Computer Science (ScienceDirect open access)
store all output artifacts under outputs/
Scope
Files to update:

 
pyproject.toml
README.md
Required Changes
1) Notebook preprocessing updates
Remove manual series standardization in preprocessing.
Delete creation and use of sales_std.
Keep raw sales as target values in series_dict.
Keep transformer_series=StandardScaler() in ForecasterRecursiveMultiSeries.
2) Add publication output directories
In notebook setup:

outputs/figures
outputs/tuning
outputs/tables
outputs/metadata
Create directories at runtime if missing.

3) Optuna + skforecast tuning (publication profile)
Add import for bayesian_search_forecaster_multiseries.
Add Optuna dependency in pyproject.toml.
Tune both lags and LightGBM hyperparameters.
Use reproducible configuration:
random seed = 42
trials = 120 default (200+ optional for camera-ready reruns)
strict time-based validation split to avoid leakage
Save tuning artifacts:
outputs/tuning/044_optuna_trials.csv
outputs/metadata/044_best_config.json
4) Final model fit and explainability
Refit final forecaster with best_lags and best_params.
Recreate X_train_sk and y_train_sk.
Run global and per-series explainability.
Increase permutation repeats to publication level (n_repeats=20).
5) Publication-quality exports
Export figure and table artifacts into outputs/:

global importance figures (PDF + PNG, 300 dpi)
per-series importance figures (PDF + PNG, 300 dpi)
SNAP/holiday heatmaps (PDF + PNG, 300 dpi)
summary bar charts (PDF + PNG, 300 dpi)
CSV tables for importance summaries
6) Documentation updates
Add Optuna dependency note in README.
Add publication profile note:
Elsevier Procedia Computer Science target
reproducibility settings
outputs folder conventions
Verification Checklist
Imports run in notebook without errors.
No remaining references to sales_std.
Forecaster fit runs with transformer_series internal scaling.
Optuna search completes and writes tuning artifacts.
Final model uses best_lags and best_params.
All publication outputs are written under outputs/ subfolders.
Figures render correctly at 300 dpi and PDFs are generated.
Constraints
Preserve existing workflow structure where possible.
Avoid modifying unrelated notebooks or source modules.
Keep deterministic seeds for reproducibility.
 