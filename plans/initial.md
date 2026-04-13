1. Introduction
1.1. The Rise of Global Forecasting Models:

Briefly introduce the concept and advantages of global models (e.g., using skforecast, Darts) over traditional single-series models (e.g., ARIMA).

Mention their power in learning cross-series patterns and handling cold-start problems.

1.2. The Interpretability Gap: The Problem of "Averaged" Explanations:

State the core problem: as models become more complex, their interpretability decreases.

Introduce the concept of post-hoc explanation and its importance for trust, debugging, and insight generation.

Critique the standard approach of applying global explanation methods (like a single PFI) to global models. Introduce the central argument: global explanations produce "averaged" insights that can be misleading.

1.3. Thesis: The Need for Conditional, Per-Series Explanations:

Propose the main solution: moving from a global explanation to a conditional one, specifically by conditioning on the series_id.

Introduce Per-Series Conditional PFI as a method to uncover heterogeneous feature effects.

State the primary hypothesis of the paper (H4: The Heterogeneous Effects Hypothesis): Global PFI will fail to identify features important to only a subset of series, while Per-Series Conditional PFI will correctly identify these varying effects.

1.4. Contributions and Paper Structure:

Summarize the key contributions:

A clear demonstration of the failure mode of Global PFI in a multi-series context.

A practical implementation and validation of Per-Series Conditional PFI.

A robust experimental design for comparing the explanatory accuracy of different methods.

Briefly outline the structure of the rest of the paper.

2. Background and Related Work
2.1. Global Models in Time Series Forecasting:

Cite foundational work and popular libraries (e.g., skforecast, lightgbm, Darts).

Briefly explain how they transform time series data into a feature matrix suitable for ML regressors.

2.2. Post-Hoc Explanation Methods:

Permutation Feature Importance (PFI): Detail the standard algorithm (Fisher et al., 2019). Crucially, explain its theoretical limitation in the presence of correlated features.

SHAP and LIME: Briefly mention these as other popular model-agnostic methods.

Model-Specific Importance: Discuss alternatives like 'Gain' in tree-based models and their own biases.

2.3. Conditional Permutation Importance:

Introduce the concept of conditional PFI as a solution to the correlation problem.

Mention automated, tree-based cs-PFI as a state-of-the-art method for discovering feature interactions.

Position our proposed Per-Series Conditional PFI as a specific, highly practical application of this conditional concept, where the conditioning is explicitly done on the series identifier—a novel focus for validation.

3. Methodology
3.1. The Global Forecasting Model:

Specify the model used: skforecast.ForecasterMultiSeries with a lightgbm.LGBMRegressor. Explain why this is a representative choice.

3.2. Explanation Methods Under Comparison:

3.2.1. Global Permutation Feature Importance (The Control):

Provide the algorithm: Shuffle each feature across the entire training matrix X and measure the drop in performance of the single trained model.

3.2.2. Per-Series Conditional PFI (The Proposed Method):

Provide the algorithm: For each unique series_id, filter the training matrix X to include only data from that series. Perform PFI on this subset using the single trained global model. This yields a separate importance ranking for each series.

3.3. Evaluation Metric for Explanatory Accuracy:

Introduce Spearman's Rank Correlation Coefficient (ρ) as the primary metric.

Explain why it is suitable for comparing ranked lists of features against a known ground truth.

4. Experimental Design
4.1. Objective: To quantitatively test the Heterogeneous Effects Hypothesis (H4).

4.2. Synthetic Data Generation Process (DGP):

Detail the DGP equations, explaining how it creates two groups of series (Group A and Group B).

Group A: y is strongly influenced by both X1 and X2.

Group B: y is only influenced by X1; X2 has zero effect.

State the ground truth: X1 is globally important, while X2's importance is heterogeneous (high for Group A, zero for Group B).

4.3. Procedure:

Generate the synthetic dataset.

Train a single skforecast global model on the entire dataset.

Apply Global PFI to the trained model to get one feature ranking.

Apply Per-Series Conditional PFI to the same model to get a feature ranking for each series.

5. Results
5.1. Global PFI Results:

Present the table or chart showing the single, averaged feature importance ranking.

Point out the key result: X2 is ranked as having low importance, below other features like lags.

5.2. Per-Series Conditional PFI Results:

Present a table or bar chart comparing the importance of X2 across all series.

Clearly show the stark difference: X2 has high importance for all Group A series and near-zero importance for all Group B series.

6. Discussion
6.1. Interpretation of Results:

Analyze why Global PFI failed: It correctly calculated the average importance of X2, but this average is a misleading statistic that masks the underlying heterogeneity.

Analyze why Per-Series Conditional PFI succeeded: By isolating explanations for each series, it provided a more accurate and complete picture of the model's learned behavior.

6.2. Implications for Practitioners (Plausibility and Actionability):

Discuss the consequences of relying on the global explanation. An analyst might incorrectly discard X2 as an unimportant feature, harming model performance for half of the series.

Discuss the value of the per-series explanation. It provides actionable insights (e.g., "We need to understand why X2 is so important for customers in Group A") and builds trust by showing that the model has learned plausible, context-specific patterns.

6.3. Limitations of the Study:

Acknowledge that the study used a synthetic dataset. While necessary for a known ground truth, future work should apply this to multiple real-world datasets.

Mention the computational cost of Per-Series PFI, as it requires running the importance calculation N times for N series.

7. Conclusion and Future Work
7.1. Conclusion:

Reiterate the main finding: Standard global explanation methods can be dangerously misleading for global forecasting models.

Conclude that Per-Series Conditional PFI is a demonstrably superior method for post-hoc explanation, providing more accurate, plausible, and actionable insights into heterogeneous feature effects.

Advocate for a shift in practice: from seeking a single global explanation to demanding per-series or conditional explanations.

7.2. Future Work:

Applying this methodology to more complex real-world datasets (e.g., retail, energy).

Developing computationally optimized versions of Per-Series PFI.

Extending the concept to other explanation methods like SHAP (Per-Series Conditional SHAP).

Developing visualization techniques to effectively explore and communicate these rich, multi-faceted explanations.