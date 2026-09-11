# Core feature-level drift detection logic for the DataDrift Monitor.


import numpy as np
import pandas as pd

from src.statistics import (
    calculate_ks_test,
    calculate_chi_square_test,
    calculate_cramers_v,
    apply_bonferroni_correction,
    has_sparse_expected_counts,
)


# Check whether a pandas Series contains numerical data.
def is_numeric_feature(series):
    """Return True when the feature contains numerical data."""

    return pd.api.types.is_numeric_dtype(series)


# Replace missing categorical values with an explicit category label.
def prepare_categorical_data(series):
    """Convert categorical data to object type and represent missing values explicitly."""

    return (
        series
        .astype("object")
        .fillna("[MISSING]")
    )


# Check whether a numerical feature contains enough valid values
# for a reliable KS test.
def validate_numerical_data(reference_series, current_series):
    """Validate numerical data before running the KS test."""

    reference_data = reference_series.dropna()
    current_data = current_series.dropna()

    # Require at least two valid values in each dataset.
    if len(reference_data) < 2:

        raise ValueError(
            "Reference dataset does not contain enough valid "
            "values for numerical drift analysis."
        )

    if len(current_data) < 2:

        raise ValueError(
            "Current dataset does not contain enough valid "
            "values for numerical drift analysis."
        )

    # Reject infinite numerical values.
    if not np.isfinite(reference_data).all():

        raise ValueError(
            "Reference dataset contains infinite or invalid "
            "values in a numerical feature."
        )

    if not np.isfinite(current_data).all():

        raise ValueError(
            "Current dataset contains infinite or invalid "
            "values in a numerical feature."
        )

    return True


# Check whether a numerical feature is constant in both datasets.
def is_constant_feature(reference_series, current_series):
    """Return True when both datasets contain only one unique numerical value."""

    reference_data = reference_series.dropna()
    current_data = current_series.dropna()

    if len(reference_data) == 0 or len(current_data) == 0:
        return False

    return (
        reference_data.nunique() == 1
        and current_data.nunique() == 1
    )


# Analyze one feature using the appropriate statistical test.
def analyze_feature(reference_series, current_series):
    """
    Analyze one feature and return its statistical drift information.

    Numerical features use the two-sample KS test.
    Categorical features use the chi-square test and Cramér's V.
    """

    # Calculate missing-value percentages before any preprocessing.
    reference_missing_pct = (
        reference_series.isna().mean() * 100
    )

    current_missing_pct = (
        current_series.isna().mean() * 100
    )


    # ---------------------------------------------------------
    # NUMERICAL FEATURE
    # ---------------------------------------------------------

    if is_numeric_feature(reference_series):

        # Validate numerical values before running the KS test.
        validate_numerical_data(
            reference_series,
            current_series,
        )

        reference_data = reference_series.dropna()
        current_data = current_series.dropna()

        # Run the KS test.
        statistic, p_value = calculate_ks_test(
            reference_data,
            current_data,
        )

        # Constant features are still analyzed by the KS test.
        # This flag is retained so the result can be interpreted
        # separately if needed in future improvements.
        constant_feature = is_constant_feature(
            reference_series,
            current_series,
        )

        return {
            "feature_type": "numerical",
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": None,
            "reference_missing_pct": reference_missing_pct,
            "current_missing_pct": current_missing_pct,
            "needs_review": False,
            "constant_feature": constant_feature,
        }


    # ---------------------------------------------------------
    # CATEGORICAL FEATURE
    # ---------------------------------------------------------

    # Treat missing categorical values as an explicit category.
    reference_data = prepare_categorical_data(
        reference_series
    )

    current_data = prepare_categorical_data(
        current_series
    )


    # A categorical feature with fewer than two combined categories
    # cannot provide a meaningful chi-square comparison.
    combined_categories = set(
        reference_data.unique()
    ).union(
        set(current_data.unique())
    )

    if len(combined_categories) < 2:

        return {
            "feature_type": "categorical",
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "reference_missing_pct": reference_missing_pct,
            "current_missing_pct": current_missing_pct,
            "needs_review": True,
            "constant_feature": True,
        }


    # Check whether expected category frequencies are too small.
    needs_review = has_sparse_expected_counts(
        reference_data,
        current_data,
    )


    # Run the chi-square test.
    statistic, p_value = calculate_chi_square_test(
        reference_data,
        current_data,
    )


    # Calculate Cramér's V.
    effect_size = calculate_cramers_v(
        reference_data,
        current_data,
    )


    # Determine whether the categorical feature is constant.
    constant_feature = (
        reference_data.nunique() == 1
        and current_data.nunique() == 1
    )


    return {
        "feature_type": "categorical",
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": effect_size,
        "reference_missing_pct": reference_missing_pct,
        "current_missing_pct": current_missing_pct,
        "needs_review": needs_review,
        "constant_feature": constant_feature,
    }


# Assign a final drift status using statistical significance
# and test reliability.
def classify_drift_status(
    raw_p_value,
    adjusted_p_value,
    alpha=0.05,
    needs_review=False,
):
    """Assign the final feature-level drift status."""

    if needs_review:

        return "Needs review"


    if adjusted_p_value < alpha:

        return "Significant drift"


    if raw_p_value < alpha:

        return "Potential drift"


    return "Stable"


# Analyze all common features, apply Bonferroni correction,
# and assign final drift statuses.
def detect_drift(
    reference_df,
    current_df,
    common_columns,
    alpha=0.05,
):
    """Detect feature-level drift across all common columns."""

    # Validate the significance level.
    if not 0 < alpha < 1:

        raise ValueError(
            "The significance level alpha must be between 0 and 1."
        )


    # Make sure there are features to analyze.
    if len(common_columns) == 0:

        raise ValueError(
            "No common features are available for drift analysis."
        )


    results = []


    # Analyze every common feature.
    for column in common_columns:

        result = analyze_feature(
            reference_df[column],
            current_df[column],
        )

        result["feature"] = column

        results.append(result)


    # Extract the raw p-values for multiple-testing correction.
    p_values = [
        result["p_value"]
        for result in results
    ]


    # Apply Bonferroni correction.
    adjusted_p_values = apply_bonferroni_correction(
        p_values
    )


    # Assign adjusted p-values and final statuses.
    for result, adjusted_p_value in zip(
        results,
        adjusted_p_values,
    ):

        result["adjusted_p_value"] = adjusted_p_value

        result["status"] = classify_drift_status(
            result["p_value"],
            adjusted_p_value,
            alpha,
            result["needs_review"],
        )


    # Return results in a consistent column order.
    return pd.DataFrame(results)[
        [
            "feature",
            "feature_type",
            "statistic",
            "p_value",
            "adjusted_p_value",
            "effect_size",
            "reference_missing_pct",
            "current_missing_pct",
            "needs_review",
            "status",
        ]
    ]


# Generate a plain-language interpretation of a feature's drift result.
def get_feature_interpretation(result):
    """Generate a human-readable explanation for a feature result."""

    feature = result["feature"]
    feature_type = result["feature_type"]
    status = result["status"]


    # Explain statistically significant numerical drift.
    if status == "Significant drift":

        if feature_type == "numerical":

            return (
                f"Significant drift detected: The distribution of "
                f"'{feature}' differs statistically between the "
                f"reference and current datasets."
            )

        return (
            f"Significant drift detected: The category proportions "
            f"of '{feature}' differ statistically between the "
            f"reference and current datasets."
        )


    # Explain potential drift before multiple-testing correction.
    if status == "Potential drift":

        return (
            f"Potential drift detected in '{feature}'. The raw "
            f"p-value indicates a possible difference, but the "
            f"Bonferroni-adjusted result does not meet the "
            f"significance threshold."
        )


    # Explain results where statistical assumptions need review.
    if status == "Needs review":

        return (
            f"Needs review: '{feature}' has sparse expected "
            f"frequencies or insufficient category variation, "
            f"so the categorical statistical result should be "
            f"interpreted cautiously."
        )


    # Explain stable features.
    return (
        f"No significant drift detected: The distribution of "
        f"'{feature}' does not show statistically significant "
        f"evidence of change between the reference and current "
        f"datasets."
    )