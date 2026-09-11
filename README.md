# DataDrift Monitor

## Statistical Data Drift Detection Dashboard

DataDrift Monitor is a local Streamlit dashboard that compares a reference
dataset with a current dataset and detects feature-level distribution changes
using statistical hypothesis tests.

The project is designed as a lightweight data monitoring MVP for understanding
whether the statistical distribution of incoming data has changed compared
with a baseline dataset.

---

## 1. Project Overview

Machine-learning systems often rely on the assumption that incoming data
remains reasonably similar to the data used during development or training.

When the distribution of incoming data changes, the model may eventually
perform differently even if the model itself has not been modified.

This project focuses on detecting such changes at the feature level.

The dashboard accepts two CSV files:

- **Reference dataset** — baseline or training-like data
- **Current dataset** — newer data that is being compared against the baseline

The application then:

1. Validates the datasets.
2. Finds common features.
3. Identifies numerical and categorical features.
4. Applies an appropriate statistical test.
5. Applies Bonferroni correction for multiple testing.
6. Assigns a drift status to each feature.
7. Displays the results using tables, metrics, charts, and explanations.

---

## 2. Project Goals

The main goals of the MVP are:

- Detect feature-level statistical distribution changes.
- Support both numerical and categorical features.
- Provide interpretable statistical results.
- Handle common data-quality problems.
- Provide visual comparison of reference and current distributions.
- Demonstrate the concepts of data drift and statistical hypothesis testing
  through an interactive dashboard.

---

## 3. Statistical Methodology

### 3.1 Numerical Features — Kolmogorov-Smirnov Test

Numerical features are analyzed using the two-sample
Kolmogorov-Smirnov (KS) test.

The KS test compares the empirical distributions of the reference and
current datasets.

#### Null hypothesis (H₀)

The reference and current distributions are the same.

#### Alternative hypothesis (H₁)

The reference and current distributions are different.

The dashboard reports:

- KS test statistic
- Raw p-value
- Bonferroni-adjusted p-value

Missing numerical values are excluded from the KS calculation and their
missing percentage is reported separately.

---

### 3.2 Categorical Features — Chi-Square Test

Categorical features are analyzed using a chi-square test based on category
frequency counts.

#### Null hypothesis (H₀)

Category proportions are independent of whether the observation belongs to
the reference or current dataset.

#### Alternative hypothesis (H₁)

Category proportions differ between the reference and current datasets.

The dashboard reports:

- Chi-square statistic
- Raw p-value
- Bonferroni-adjusted p-value

Missing categorical values are treated as an explicit:

`[MISSING]`

category.

---

### 3.3 Cramér's V

Cramér's V is used as an effect-size measure for categorical features.

It provides information about the magnitude of the categorical distribution
difference in addition to statistical significance.

---

### 3.4 Bonferroni Correction

Multiple features are tested simultaneously.

To reduce the chance of false-positive significance caused by multiple
statistical tests, Bonferroni correction is applied.

The adjusted significance threshold is:

`α / m`

where:

- `α` = selected significance level
- `m` = number of tested features

The default significance level in the dashboard is:

`α = 0.05`

The user can change this value from the sidebar.

---

## 4. Drift Statuses

Each feature receives one of four statuses.

### Stable

The Bonferroni-adjusted p-value is greater than or equal to the selected
significance level.

There is no statistically significant evidence of distribution change.

### Potential Drift

The raw p-value is below the significance level, but the
Bonferroni-adjusted p-value does not remain significant.

This indicates a possible difference that does not remain statistically
significant after multiple-testing correction.

### Significant Drift

The Bonferroni-adjusted p-value is below the selected significance level.

The feature shows statistically significant evidence of distribution change
after multiple-testing correction.

### Needs Review

The statistical comparison may not be reliable because of a data or
assumption issue.

For example, categorical features with sparse expected frequencies may be
flagged for review.

---

## 5. Dashboard Features

The Streamlit dashboard provides:

- Reference CSV upload
- Current CSV upload
- Configurable significance level
- Dataset validation
- Dataset summary
- Overall drift summary
- Feature-level results table
- KS statistics for numerical features
- Chi-square statistics for categorical features
- Cramér's V for categorical features
- Raw and adjusted p-values
- Missing-value percentages
- Feature-level interpretation
- Numerical distribution comparison
- Categorical distribution comparison
- Drift summary chart
- Full results CSV download
- Statistical methodology explanation

---

## 6. Project Structure

```text
DriftDetection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── pytest.ini
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── validator.py
│   ├── drift_detector.py
│   ├── statistics.py
│   └── visualization.py
│
├── data/
│   └── sample/
│       ├── reference.csv
│       ├── current.csv
│       ├── current_stable.csv
│       ├── current_mild.csv
│       ├── current_categorical.csv
│       ├── current_missing.csv
│       ├── current_sparse.csv
│       ├── current_constant.csv
│       └── current_mixed.csv
│
└── tests/
    ├── test_validation.py
    └── test_drift.py