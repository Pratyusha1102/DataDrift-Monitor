# Test Phase 2 descriptive findings and HTML report generation.

import pandas as pd

from src.drift_detector import detect_drift
from src.insights import (
    add_phase_two_insights,
    analyze_drift_relationships,
    build_html_report,
    categorical_change_details,
)


# Verify categorical additions and removals are reported correctly.
def test_category_changes_identify_new_and_removed_values():
    added, removed, details = categorical_change_details(
        pd.Series(["Basic", "Basic", "Pro"]),
        pd.Series(["Pro", "Enterprise", "Enterprise"]),
    )

    assert added == ["Enterprise"]
    assert removed == ["Basic"]
    assert "change_percentage_points" in details


# Verify feature results gain descriptive Phase 2 fields without a score.
def test_phase_two_results_include_explanation_and_severity():
    reference = pd.DataFrame(
        {
            "age": list(range(20)),
            "plan": ["Basic"] * 15 + ["Pro"] * 5,
        }
    )

    current = pd.DataFrame(
        {
            "age": list(range(100, 120)),
            "plan": ["Enterprise"] * 15 + ["Pro"] * 5,
        }
    )

    results = add_phase_two_insights(
        detect_drift(reference, current, ["age", "plan"]),
        reference,
        current,
    )

    assert {
        "severity",
        "what_changed",
        "new_categories",
        "removed_categories",
    }.issubset(results.columns)
    assert "impact_score" not in results.columns
    assert (
        results.loc[
            results["feature"] == "plan",
            "new_categories",
        ].iloc[0]
        == "Enterprise"
    )


# Verify substantial numerical correlation changes are reported.
def test_relationship_analysis_detects_changed_correlation():
    reference = pd.DataFrame({"a": range(20), "b": range(20)})
    current = pd.DataFrame(
        {"a": range(20), "b": list(range(19, -1, -1))}
    )

    relationships = analyze_drift_relationships(
        reference,
        current,
        ["a", "b"],
    )

    assert len(relationships) == 1
    assert relationships.loc[0, "change"] < 0


# Verify the downloadable report contains feature-level drift content.
def test_html_report_contains_drift_content():
    results = pd.DataFrame(
        {
            "feature": ["age"],
            "feature_type": ["numerical"],
            "status": ["Stable"],
            "severity": ["None"],
            "what_changed": ["Median did not change."],
            "new_categories": [""],
            "removed_categories": [""],
        }
    )

    report = build_html_report(results, pd.DataFrame(), 0.05)

    assert "Data Drift Report" in report
    assert "age" in report
