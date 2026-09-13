# Statistical tests and correction methods used by the DataDrift Monitor.

import numpy as np

# Calculate the two-sample Kolmogorov-Smirnov test for numerical data.
def calculate_ks_test(reference_data, current_data):
    from scipy.stats import ks_2samp

    """Calculate the two-sample Kolmogorov-Smirnov test."""

    reference_data = np.asarray(reference_data)
    current_data = np.asarray(current_data)

    # Require enough observations in both datasets for the test.
    if len(reference_data) < 2 or len(current_data) < 2:
        raise ValueError(
            "The KS test requires at least two valid numerical "
            "values in each dataset."
        )

    # Reject invalid numerical values before running the test.
    if not np.isfinite(reference_data).all():
        raise ValueError(
            "Reference numerical data contains invalid values."
        )

    if not np.isfinite(current_data).all():
        raise ValueError(
            "Current numerical data contains invalid values."
        )

    statistic, p_value = ks_2samp(
        reference_data,
        current_data,
    )

    return float(statistic), float(p_value)


# Calculate the chi-square test using category frequency counts.
def calculate_chi_square_test(reference_data, current_data):
    from scipy.stats import chi2_contingency

    reference_counts = reference_data.value_counts()
    current_counts = current_data.value_counts()

    # Create one combined category index so both datasets
    # use exactly the same category order.
    categories = reference_counts.index.union(
        current_counts.index
    )

    reference_counts = reference_counts.reindex(
        categories,
        fill_value=0,
    )

    current_counts = current_counts.reindex(
        categories,
        fill_value=0,
    )

    contingency_table = np.array(
        [
            reference_counts.values,
            current_counts.values,
        ],
        dtype=float,
    )

    # A chi-square comparison requires at least two categories.
    if contingency_table.shape[1] < 2:
        raise ValueError(
            "The chi-square test requires at least two categories."
        )

    # Both datasets must contain at least one observation.
    if contingency_table.sum() == 0:
        raise ValueError(
            "Categorical data contains no valid observations."
        )

    statistic, p_value, _, _ = chi2_contingency(
        contingency_table
    )

    return float(statistic), float(p_value)


# Calculate Cramér's V as the effect-size measure for categorical drift.
def calculate_cramers_v(reference_data, current_data):
    """Calculate Cramér's V for categorical data."""
    from scipy.stats import chi2_contingency

    reference_counts = reference_data.value_counts()
    current_counts = current_data.value_counts()

    categories = reference_counts.index.union(
        current_counts.index
    )

    reference_counts = reference_counts.reindex(
        categories,
        fill_value=0,
    )

    current_counts = current_counts.reindex(
        categories,
        fill_value=0,
    )

    contingency_table = np.array(
        [
            reference_counts.values,
            current_counts.values,
        ],
        dtype=float,
    )

    # Cramér's V is not meaningful with fewer than two categories.
    if contingency_table.shape[1] < 2:
        return 0.0

    total = contingency_table.sum()

    if total <= 1:
        return 0.0

    statistic, _, _, _ = chi2_contingency(
        contingency_table
    )

    rows, columns = contingency_table.shape

    # Calculate the uncorrected phi-squared value.
    phi_squared = statistic / total

    # Apply the bias correction proposed for Cramér's V.
    correction = (
        (columns - 1) * (rows - 1)
    ) / (total - 1)

    phi_squared_corrected = max(
        0.0,
        phi_squared - correction,
    )

    rows_corrected = rows - (
        (rows - 1) ** 2 / (total - 1)
    )

    columns_corrected = columns - (
        (columns - 1) ** 2 / (total - 1)
    )

    denominator = min(
        rows_corrected - 1,
        columns_corrected - 1,
    )

    # Avoid division by zero for degenerate tables.
    if denominator <= 0:
        return 0.0

    return float(
        np.sqrt(
            phi_squared_corrected / denominator
        )
    )


# Apply Bonferroni correction to a collection of p-values.
def apply_bonferroni_correction(p_values):
    """Apply Bonferroni correction to a collection of p-values."""

    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return []

    adjusted_p_values = []

    for p_value in p_values:

        # Protect against invalid p-values.
        if not np.isfinite(p_value):
            adjusted_p_values.append(1.0)
            continue

        adjusted_value = min(
            float(p_value) * number_of_tests,
            1.0,
        )

        adjusted_p_values.append(
            adjusted_value
        )

    return adjusted_p_values


# Check whether categorical expected frequencies are too small.
def has_sparse_expected_counts(
    reference_data,
    current_data,
    threshold=5,
):
    """Check whether a categorical comparison has sparse expected frequencies."""
    from scipy.stats import chi2_contingency

    reference_counts = reference_data.value_counts()
    current_counts = current_data.value_counts()

    categories = reference_counts.index.union(
        current_counts.index
    )

    reference_counts = reference_counts.reindex(
        categories,
        fill_value=0,
    )

    current_counts = current_counts.reindex(
        categories,
        fill_value=0,
    )

    contingency_table = np.array(
        [
            reference_counts.values,
            current_counts.values,
        ],
        dtype=float,
    )

    # Fewer than two categories cannot produce a meaningful
    # chi-square comparison.
    if contingency_table.shape[1] < 2:
        return True

    # Calculate expected frequencies without modifying the
    # statistical method used by the detector.
    try:

        _, _, _, expected = chi2_contingency(
            contingency_table
        )

    except ValueError:

        return True

    return bool(
        (expected < threshold).any()
    )