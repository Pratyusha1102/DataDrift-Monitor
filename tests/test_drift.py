# Test the statistical drift detection functions.


import pandas as pd

from src.statistics import (
    calculate_ks_test,
    calculate_chi_square_test,
    calculate_cramers_v,
    apply_bonferroni_correction,
)

from src.drift_detector import (
    is_numeric_feature,
    analyze_feature,
    detect_drift,
    classify_drift_severity,
)

# Verify that numerical data is correctly identified.
def test_is_numeric_feature():
    series = pd.Series([10, 20, 30])

    assert is_numeric_feature(series) is True


# Verify that categorical data is not classified as numerical.
def test_is_numeric_feature_categorical():
    series = pd.Series(["A", "B", "C"])

    assert is_numeric_feature(series) is False


# Verify that the KS test detects clearly different numerical distributions.
def test_ks_test_detects_difference():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = pd.Series([10, 11, 12, 13, 14])

    statistic, p_value = calculate_ks_test(reference, current)

    assert statistic == 1.0
    assert p_value < 0.05


# Verify that identical numerical distributions produce no KS drift signal.
def test_ks_test_identical_data():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = pd.Series([1, 2, 3, 4, 5])

    statistic, p_value = calculate_ks_test(reference, current)

    assert statistic == 0.0
    assert p_value == 1.0


# Verify that the Bonferroni correction increases p-values appropriately.
def test_bonferroni_correction():
    p_values = [0.001, 0.02, 0.04, 0.3]

    adjusted = apply_bonferroni_correction(p_values)

    assert adjusted == [
        0.004,
        0.08,
        0.16,
        1.0,
    ]


# Verify that categorical data can be analyzed with the chi-square test.
def test_chi_square_test():
    reference = pd.Series(["A", "A", "B", "B", "C"])

    current = pd.Series(["A", "B", "B", "C", "C"])

    statistic, p_value = calculate_chi_square_test(reference, current)

    assert statistic >= 0
    assert 0 <= p_value <= 1


# Verify that Cramér's V returns a valid effect-size value.
def test_cramers_v():
    reference = pd.Series(["A", "A", "B", "B", "C"])

    current = pd.Series(["A", "B", "B", "C", "C"])

    effect_size = calculate_cramers_v(reference, current)

    assert 0 <= effect_size <= 1


# Verify that the complete detector identifies numerical drift.
def test_detect_drift_identifies_significant_drift():
    reference_df = pd.DataFrame({"age": [20, 21, 22, 23, 24]})

    current_df = pd.DataFrame({"age": [30, 31, 32, 33, 34]})

    results = detect_drift(reference_df, current_df, ["age"])

    assert results.loc[0, "status"] == "Significant drift"


# Verify that sparse categorical data is flagged for review.
def test_detect_drift_flags_sparse_categories():
    reference_df = pd.DataFrame({"education": ["A", "A", "B", "B", "C"]})

    current_df = pd.DataFrame({"education": ["A", "B", "B", "C", "C"]})

    results = detect_drift(reference_df, current_df, ["education"])

    assert results.loc[0, "status"] == "Needs review"

# Verify that a large numerical KS statistic is classified as high severity.
def test_classify_numerical_high_severity():
    assert classify_drift_severity(
        status="Significant drift",
        feature_type="numerical",
        statistic=0.60,
        effect_size=None,
    ) == "High"


# Verify that a moderate numerical KS statistic is classified as medium severity.
def test_classify_numerical_medium_severity():
    assert classify_drift_severity(
        status="Significant drift",
        feature_type="numerical",
        statistic=0.30,
        effect_size=None,
    ) == "Medium"


# Verify that a small numerical KS statistic is classified as low severity.
def test_classify_numerical_low_severity():
    assert classify_drift_severity(
        status="Significant drift",
        feature_type="numerical",
        statistic=0.10,
        effect_size=None,
    ) == "Low"


# Verify that a large categorical Cramér's V is classified as high severity.
def test_classify_categorical_high_severity():
    assert classify_drift_severity(
        status="Significant drift",
        feature_type="categorical",
        statistic=18.0,
        effect_size=0.60,
    ) == "High"


# Verify that a moderate categorical Cramér's V is classified as medium severity.
def test_classify_categorical_medium_severity():
    assert classify_drift_severity(
        status="Significant drift",
        feature_type="categorical",
        statistic=12.0,
        effect_size=0.35,
    ) == "Medium"


# Verify that a small categorical Cramér's V is classified as low severity.
def test_classify_categorical_low_severity():
    assert classify_drift_severity(
        status="Significant drift",
        feature_type="categorical",
        statistic=5.0,
        effect_size=0.10,
    ) == "Low"


# Verify that potential drift receives a review classification.
def test_classify_potential_drift_for_review():
    assert classify_drift_severity(
        status="Potential drift",
        feature_type="numerical",
        statistic=0.40,
        effect_size=None,
    ) == "Review"


# Verify that stable features receive no drift severity.
def test_classify_stable_feature():
    assert classify_drift_severity(
        status="Stable",
        feature_type="numerical",
        statistic=0.01,
        effect_size=None,
    ) == "None"


# Verify that data-quality problems are explicitly marked for review.
def test_classify_needs_review():
    assert classify_drift_severity(
        status="Needs review",
        feature_type="categorical",
        statistic=None,
        effect_size=None,
    ) == "Needs review"

# Verify that detect_drift includes severity classifications in its results.
def test_detect_drift_includes_severity():
    reference = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    )

    current = pd.DataFrame(
        {
            "age": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        }
    )

    results = detect_drift(reference, current, ["age"])

    assert "severity" in results.columns
    assert results.loc[0, "severity"] in {
        "High",
        "Medium",
        "Low",
        "Review",
        "None",
        "Needs review",
    }
