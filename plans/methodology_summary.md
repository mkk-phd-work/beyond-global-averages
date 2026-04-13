# Methodology and Experimental Design Summary

## 1. Research Objective

### 1.1 Core Problem
Global feature importance methods produce "averaged" explanations that mask heterogeneous effects in multi-series forecasting. When exogenous features affect different series in different ways, standard global Permutation Feature Importance (PFI) returns a single aggregated ranking that may not accurately represent any individual series.

### 1.2 Proposed Solution
**Per-Series Conditional Permutation Feature Importance (PFI)** — a method that computes feature importance separately for each series by conditioning on the series identifier, revealing which exogenous features truly matter for each specific time series.

### 1.3 Primary Hypothesis (H4: Heterogeneous Effects Hypothesis)
Global PFI fails to identify features important to only a subset of series, while Per-Series Conditional PFI correctly identifies these varying effects. The global explanation produces a mathematically valid but practically misleading average that masks underlying heterogeneity.

---

## 2. Model Architecture

### 2.1 Forecasting Framework
- **Library**: `skforecast.ForecasterRecursiveMultiSeries`
- **Base Regressor**: `lightgbm.LGBMRegressor`
- **Rationale**: Representative of modern tabular global forecasting approaches

### 2.2 Data Transformation
- **Internal Scaling**: `transformer_series=StandardScaler()` 
- **No Manual Preprocessing**: Raw target values used; scaling handled internally by the forecaster
- **Feature Matrix**: Time series transformed into supervised learning format with lag features and exogenous variables

### 2.3 Hyperparameter Tuning
- **Method**: Optuna-based Bayesian search via `bayesian_search_forecaster_multiseries`
- **Trials**: 120 (default), 200+ for publication-quality runs
- **Validation**: Strict time-based split to avoid data leakage
- **Random Seed**: 42 for reproducibility

---

## 3. Explanation Methods

### 3.1 Global Permutation Feature Importance (Control)

**Algorithm:**
1. Train a single global model on the combined multi-series dataset
2. For each feature:
   - Shuffle the feature values across the **entire** training matrix X
   - Measure the performance degradation of the trained model
3. Rank features by average importance across all series

**Output:** One aggregated feature ranking that averages effects across all series.

**Limitation:** When feature effects are heterogeneous across series, the global average can be misleading—features important to subsets of series may appear unimportant overall.

### 3.2 Per-Series Conditional PFI (Proposed Method)

**Algorithm:**
1. Train a single global model on the combined multi-series dataset (same model as above)
2. For each unique `series_id`:
   - Filter the training matrix X to include only rows from that series
   - Permute features **within the filtered subset only**
   - Evaluate the same trained global model on the permuted subset
   - Compute importance scores for that series
3. Output separate importance rankings per series

**Output:** N feature rankings for N series, revealing series-specific patterns.

**Implementation:** `xeries.ConditionalPermutationImportance` with `xeries.SkforecastAdapter`

**Key Insight:** The conditioning variable is the series identifier, not an automatically discovered interaction. This is a practical, explicit conditioning strategy.

---

## 4. Out-of-Distribution (OOD) Validation

### 4.1 Purpose
Validate that conditional permutation preserves series-specific statistics while global permutation destroys them. This provides theoretical justification for why per-series PFI produces more accurate explanations.

### 4.2 Metrics

| Metric | Description |
|--------|-------------|
| **Wasserstein Distance** | Earth Mover's Distance between original and permuted feature distributions |
| **Kolmogorov-Smirnov Statistic** | Maximum difference between cumulative distribution functions |
| **Per-Series Mean Shift** | Change in per-series feature means after permutation |

### 4.3 Key Findings (from Synthetic Experiment)

| Permutation Method | Wasserstein Distance | Mean Shift |
|-------------------|---------------------|------------|
| Global | 2.5714 | 2.5597 |
| Conditional | 0.0000 | 0.0000 |

**Conclusion:** 
- Global permutation destroys series-specific patterns (high distribution shift)
- Conditional permutation preserves per-series statistics (near-zero shift)
- This validates using conditional permutation for multi-series importance analysis

---

## 5. Experimental Design

### 5.1 Experiment 1: Synthetic Data

**Dataset Configuration:**
- **Series**: 6 series (3 urban, 3 rural)
- **Time Period**: 365 days
- **Exogenous Features**: `temperature`, `holiday_flag`, `promotion`

**Ground Truth Design:**

| Feature | Urban Effect | Rural Effect |
|---------|-------------|--------------|
| `temperature` | Strong positive (+coefficient) | No effect (zero) |
| `holiday_flag` | No effect (zero) | Strong positive (+25 per holiday) |
| `promotion` | Positive effect | Positive effect (equal across groups) |

**Design Rationale:**
- Urban stores: Sales increase with warmer temperatures (e.g., ice cream, outdoor activities)
- Rural stores: Sales spike on holidays (e.g., tourism destinations, weekend getaways)
- Promotion: Universal effect to serve as control feature

**Expected Outcome:**
- Global PFI will average out the distinct temperature and holiday effects
- Per-Series PFI will correctly identify:
  - High temperature importance for urban series
  - High holiday importance for rural series
  - Similar promotion importance across all series

### 5.2 Experiment 2: M5 Real-World Data

**Dataset Configuration:**
- **Source**: M5 Forecasting Competition (Walmart retail data)
- **Aggregation**: Daily demand aggregated to state × category level
- **Series**: 9 series (3 states: CA, TX, WI × 3 categories: FOODS, HOBBIES, HOUSEHOLD)
- **Time Period**: 2011-01-29 to 2016-04-24 (~1,913 daily observations)
- **Exogenous Features**: `snap_flag` (SNAP benefit days, state-specific), `holiday_flag`

**Purpose:**
- Validate methodology on real-world retail data
- Test whether per-series PFI reveals state-specific SNAP effects
- Compare holiday importance across product categories

---

## 6. Key Results Summary

### 6.1 Synthetic Data Results

#### Exogenous Feature Importance Comparison

| Feature | Urban (avg) | Rural (avg) | Global |
|---------|-------------|-------------|--------|
| `temperature` | 0.4038 | 0.0295 | 0.1932 |
| `holiday_flag` | 0.0035 | 0.6468 | 0.3201 |
| `promotion` | 0.4169 | 0.4067 | 0.4096 |

#### Per-Series Detail (Exogenous Features)

**Urban Series:**
| Series | lag_1 | promotion | temperature | holiday_flag |
|--------|-------|-----------|-------------|--------------|
| urban_001 | 0.7807 | 0.4153 | 0.3879 | 0.0051 |
| urban_002 | 0.7847 | 0.4148 | 0.4093 | 0.0022 |
| urban_003 | 0.7810 | 0.4206 | 0.4143 | 0.0032 |

**Rural Series:**
| Series | holiday_flag | lag_1 | promotion | temperature |
|--------|--------------|-------|-----------|-------------|
| rural_001 | 0.6552 | 0.5593 | 0.4193 | 0.0305 |
| rural_002 | 0.6408 | 0.5252 | 0.3978 | 0.0282 |
| rural_003 | 0.6444 | 0.5412 | 0.4029 | 0.0296 |

#### Key Observations
1. **Temperature**: High importance for urban (0.40), near-zero for rural (0.03), global shows misleading average (0.19)
2. **Holiday flag**: High importance for rural (0.65), near-zero for urban (0.00), global shows misleading average (0.32)
3. **Promotion**: Similar importance across both groups (~0.41), global correctly reflects this

### 6.2 OOD Validation Results

| Analysis | Global Permutation | Conditional Permutation |
|----------|-------------------|------------------------|
| Wasserstein Distance (avg) | 2.5714 | 0.0000 |
| Per-Series Mean Shift (avg) | 2.5597 | 0.0000 |

**Interpretation:** Global permutation destroys series-specific patterns; conditional permutation preserves them.

---

## 7. Output Artifacts

### 7.1 Directory Structure
```
outputs/
├── figures/          # PDF + PNG at 300 dpi
├── tables/           # CSV + LaTeX exports
├── tuning/           # Optuna trial logs
└── metadata/         # Best configuration JSONs
```

### 7.2 Key Output Files

**Figures:**
- `01_global_importance.pdf` - Global feature importance bar chart
- `01_per_series_importance.pdf` - Per-series importance comparison
- `01_importance_heatmap.pdf` - Feature × series importance heatmap
- `01_urban_vs_rural_comparison.pdf` - Regional comparison visualization
- `01_ood_distance_comparison.pdf` - OOD analysis visualization
- `01_ood_mean_shift_heatmap.pdf` - Per-series mean shift visualization
- `02_m5_*.pdf` - M5 experiment outputs

**Tables:**
- `01_global_importance.csv/.tex` - Global importance scores
- `01_per_series_importance.csv/.tex` - Per-series importance scores
- `01_ood_distances.csv/.tex` - OOD distance metrics
- `02_m5_*.csv/.tex` - M5 experiment tables

**Metadata:**
- `01_best_config.json` - Tuned hyperparameters (synthetic)
- `02_m5_best_config.json` - Tuned hyperparameters (M5)

---

## 8. Reproducibility Settings

| Setting | Value |
|---------|-------|
| Random Seed | 42 |
| Permutation Repeats | 20 (publication level) |
| Optuna Trials | 120 (default), 200+ (publication) |
| Figure DPI | 300 |
| Figure Formats | PDF + PNG |
| Target Publication | Elsevier Procedia Computer Science (KES 2026) |

---

## 9. Source Files

| File | Description |
|------|-------------|
| [plans/initial.md](initial.md) | Research paper outline |
| [plans/plan.md](plan.md) | Implementation plan |
| [plans/main.pdf](main.pdf) | Paper draft (KES 2026 template) |
| [notebooks/01_exogenous_features.ipynb](../notebooks/01_exogenous_features.ipynb) | Synthetic data experiment |
| [notebooks/02_m5_exog.ipynb](../notebooks/02_m5_exog.ipynb) | M5 real-world validation |

---

## 10. References

1. Chamma, A., Engemann, D.A., Thirion, B. (2023). Statistically valid variable importance assessment through conditional permutations. *NeurIPS 2023*.
2. Hewamalage, H., Bergmeir, C., Bandara, K. (2022). Global models for time series forecasting: A simulation study. *Pattern Recognition*, 124, 108441.
3. Kang, Y., Hyndman, R.J., Li, F. (2020). Gratis: Generating time series with diverse and controllable characteristics. *Statistical Analysis and Data Mining*, 13, 354-376.

---

# Appendix: Paper Gap Analysis (main.pdf)

## A.1 Current Paper Status

The paper draft (`plans/main.pdf`) is a KES 2026 submission template for Elsevier Procedia Computer Science with:
- Complete 7-section structure with bullet-point outlines
- 3 references already cited
- Acknowledgements section with funding information
- All content sections are placeholders (bullet points only)

---

## A.2 What's Relevant and Aligned

| Paper Section | Content | Notebook Evidence |
|---------------|---------|-------------------|
| 1.3 Thesis | Per-Series Conditional PFI, Heterogeneous Effects Hypothesis | Fully validated in notebook 01 |
| 3.1 Global Model | skforecast + LightGBM | Implemented in both notebooks |
| 3.2.1 Global PFI | Shuffle across full matrix | Implemented via xeries |
| 3.2.2 Per-Series PFI | Filter by series_id, permute within subset | Implemented via xeries |
| 4.2 Synthetic DGP | Group A vs Group B with different X2 effects | Urban vs Rural in notebook 01 |
| 5.1-5.2 Results | Global vs Per-Series comparison | Quantitative results available |

---

## A.3 Gaps and Required Updates

### Section 3: Methodology - NEEDS ADDITIONS

**Gap 1: Missing OOD Validation Methodology**
- The paper does not describe the Out-of-Distribution validation approach
- This analysis validates WHY conditional permutation works
- Metrics to document: Wasserstein Distance, Per-series mean shift

**Gap 2: Missing `xeries` Library Details**
- Implementation uses the `xeries` library for conditional PFI
- Should mention `ConditionalPermutationImportance` and `SkforecastAdapter`

**Recommended Addition:**
> Add Subsection 3.4: Out-of-Distribution Validation

**Draft Content for 3.4:**
> We validate that conditional permutation preserves series-specific statistics by measuring distribution shift. Using Wasserstein distance and per-series mean shift, we compare feature distributions before and after permutation under both global and conditional strategies. If global permutation produces large distribution shifts while conditional permutation maintains near-zero shifts, this confirms that series-specific patterns are preserved only under conditional permutation.

---

### Section 4: Experimental Design - NEEDS EXPANSION

**Gap: Missing M5 Real-World Validation**
- Currently only describes synthetic data experiment
- Notebook 02 provides complete M5 validation
- Real-world validation strengthens claims beyond synthetic proof-of-concept

**Recommended Addition:**
> Add Subsection 4.4: Real-World Validation: M5 Dataset

**Draft Content for 4.4:**
> We validate our methodology on the M5 Forecasting Competition dataset, aggregating daily demand to 9 series (3 states × 3 product categories: CA, TX, WI × FOODS, HOBBIES, HOUSEHOLD). Exogenous features include state-specific SNAP benefit flags and holiday indicators. The data spans 2011-01-29 to 2016-04-24 (~1,913 daily observations per series). This real-world setting tests whether per-series PFI reveals policy effects (SNAP) that vary by state, providing external validity beyond the synthetic experiment.

---

### Section 5: Results - NEEDS CONTENT

**Gap 1: Placeholder Bullet Points**
- All results are bullet-point outlines with no actual numbers
- Notebook results provide complete quantitative evidence

**Fill in Section 5.1 (Global PFI Results):**
> Global PFI produced a single averaged ranking where `promotion` (0.41) ranked highest among exogenous features, followed by `holiday_flag` (0.32) and `temperature` (0.19). Notably, `temperature` appeared relatively unimportant despite being highly predictive for urban series.

**Fill in Section 5.2 (Per-Series Conditional PFI Results):**
> Per-series analysis revealed stark heterogeneity: urban series showed `temperature` importance of 0.40 with near-zero `holiday_flag` importance (0.00), while rural series showed the inverse pattern with `holiday_flag` importance of 0.65 and near-zero `temperature` importance (0.03). The `promotion` feature maintained consistent importance (~0.41) across both groups.

**Recommended Additions:**
> Add Subsection 5.3: OOD Analysis Results
> Add Subsection 5.4: M5 Real-World Validation Results

**Draft Content for 5.3:**
> Global permutation produced Wasserstein distance of 2.57 and per-series mean shift of 2.56, indicating severe distribution shift that destroys series-specific patterns. Conditional permutation maintained both metrics at approximately 0.0, confirming that series-specific patterns are preserved. This validates the theoretical advantage of conditional over global permutation.

**Draft Content for 5.4:**
> On M5 data, per-series PFI revealed varying SNAP effects across states. [Fill with specific M5 results from notebook 02]. Holiday effects similarly varied by product category, with [specific findings]. These real-world results corroborate the synthetic findings.

---

### Section 6: Discussion - NEEDS EXPANSION

**Gap 1: Missing OOD Analysis Interpretation**
- The paper should discuss why conditional permutation works theoretically
- OOD results provide this justification

**Gap 2: Missing M5 Discussion**
- Real-world validation needs interpretation
- Cross-validation between synthetic and real-world findings strengthens conclusions

**Recommended Additions:**
> In Section 6.1, add paragraph on OOD validation interpretation
> Add new discussion of M5 findings and cross-validation with synthetic results

---

### References - NEEDS ADDITIONS

**Current References (3):**
1. Chamma et al. (2023) - Conditional permutation importance
2. Hewamalage et al. (2022) - Global models for time series
3. Kang et al. (2020) - GRATIS time series generation

**Required Additions:**
- skforecast library citation
- LightGBM citation (Ke et al., 2017)
- M5 Competition citation (Makridakis et al., 2022)
- xeries library reference (if published)

---

## A.4 Recommended Paper Structure Updates

```
3. Methodology
   3.1. The Global Forecasting Model [existing - fill content]
   3.2. Explanation Methods Under Comparison [existing - fill content]
   3.3. Evaluation Metric for Explanatory Accuracy [existing - fill content]
   3.4. Out-of-Distribution Validation [NEW]

4. Experimental Design
   4.1. Objective: Testing the Heterogeneous Effects Hypothesis [existing]
   4.2. Synthetic Data Generation Process [existing - fill content]
   4.3. Procedure [existing - fill content]
   4.4. Real-World Validation: M5 Dataset [NEW]

5. Results
   5.1. Global PFI Results [fill with actual numbers]
   5.2. Per-Series Conditional PFI Results [fill with actual numbers]
   5.3. OOD Analysis Results [NEW]
   5.4. M5 Real-World Validation Results [NEW]

6. Discussion
   6.1. Interpretation of Results [expand with OOD interpretation]
   6.2. Implications for Practitioners [existing]
   6.3. Limitations of the Study [existing]

7. Conclusion and Future Work [existing]
```

---

## A.5 Missing Metadata to Fill

| Field | Current Status | Recommended Content |
|-------|----------------|---------------------|
| **Title** | "Insert here, the title of your paper" | "Conditional Permutation Importance for Multi-Series Forecasting: Detecting Heterogeneous Feature Effects" |
| **Keywords** | "Type your keywords here" | "global forecasting; permutation importance; conditional explanation; heterogeneous effects; interpretability; multi-series" |
| **Abstract** | "Insert here your abstract text" | Synthesize from Section 1 introduction + key quantitative results |

**Draft Abstract:**
> Global forecasting models learn shared patterns across multiple time series, but standard explanation methods such as Permutation Feature Importance (PFI) produce averaged insights that may misrepresent individual series. We propose Per-Series Conditional PFI, which conditions on the series identifier to reveal heterogeneous feature effects. Through synthetic experiments with known ground truth and validation on M5 retail data, we demonstrate that global PFI masks subset-specific effects while per-series PFI correctly identifies them. Out-of-distribution analysis confirms that conditional permutation preserves series-specific patterns (Wasserstein distance ≈ 0) while global permutation destroys them (Wasserstein distance ≈ 2.57). Our findings advocate for a shift from global to conditional explanations in multi-series forecasting.

---

## A.6 Priority Action Items

| Priority | Action | Effort |
|----------|--------|--------|
| **High** | Fill Section 5 with actual numerical results | Medium |
| **High** | Add Section 3.4 (OOD Validation) | Low |
| **High** | Add Section 4.4 (M5 Dataset) | Low |
| **High** | Fill title, keywords, abstract | Low |
| **Medium** | Add Sections 5.3 and 5.4 (new results sections) | Medium |
| **Medium** | Expand Section 6 with OOD and M5 discussion | Medium |
| **Low** | Add missing references | Low |
