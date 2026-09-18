# Build descriptive Phase 2 findings without changing drift test results.

from html import escape

import pandas as pd


# Identify categorical additions, removals, and the largest share changes.
def categorical_change_details(reference, current):
    """Return new and removed categories plus category proportion changes."""

    reference_values = (
        reference.astype("object")
        .fillna("[MISSING]")
        .astype(str)
    )

    current_values = (
        current.astype("object")
        .fillna("[MISSING]")
        .astype(str)
    )

    reference_categories = set(reference_values.unique())
    current_categories = set(current_values.unique())
    categories = sorted(reference_categories | current_categories)

    reference_pct = (
        reference_values.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        * 100
    )

    current_pct = (
        current_values.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        * 100
    )

    changes = pd.DataFrame(
        {
            "category": categories,
            "reference_pct": reference_pct.round(2).values,
            "current_pct": current_pct.round(2).values,
        }
    )

    changes["change_percentage_points"] = (
        changes["current_pct"] - changes["reference_pct"]
    ).round(2)

    changes = changes.sort_values(
        "change_percentage_points",
        key=lambda values: values.abs(),
        ascending=False,
        ignore_index=True,
    )

    return (
        sorted(current_categories - reference_categories),
        sorted(reference_categories - current_categories),
        changes,
    )


# Describe the numerical median change without assigning business meaning.
def numerical_change_summary(reference, current):
    """Return a concise, descriptive numerical median comparison."""

    reference_median = float(reference.dropna().median())
    current_median = float(current.dropna().median())
    difference = current_median - reference_median

    if reference_median == 0:
        percentage = "percentage change unavailable from a zero baseline"
    else:
        percentage = (
            f"{difference / abs(reference_median) * 100:+.1f}%"
        )

    if difference > 0:
        direction = "increased"
    elif difference < 0:
        direction = "decreased"
    else:
        direction = "did not change"

    return (
        f"Median {direction} from {reference_median:.3g} "
        f"to {current_median:.3g} ({difference:+.3g}; {percentage})."
    )


# Add plain-language descriptive findings to existing feature-level results.
def add_phase_two_insights(results, reference, current):
    """Add What Changed and categorical category details to drift results."""

    enriched = results.copy()
    explanations = []
    new_categories = []
    removed_categories = []

    for _, result in enriched.iterrows():
        feature = result["feature"]

        if result["feature_type"] == "numerical":
            explanations.append(
                numerical_change_summary(
                    reference[feature],
                    current[feature],
                )
            )
            new_categories.append("")
            removed_categories.append("")

        else:
            added, removed, details = categorical_change_details(
                reference[feature],
                current[feature],
            )

            largest_change = details.iloc[0]
            explanation = (
                f"'{largest_change['category']}' changed by "
                f"{largest_change['change_percentage_points']:+.2f} "
                "percentage points."
            )

            if added:
                explanation += " New categories: " + ", ".join(added) + "."

            if removed:
                explanation += (
                    " Removed categories: " + ", ".join(removed) + "."
                )

            explanations.append(explanation)
            new_categories.append(", ".join(added))
            removed_categories.append(", ".join(removed))

    enriched["what_changed"] = explanations
    enriched["new_categories"] = new_categories
    enriched["removed_categories"] = removed_categories

    return enriched


# Find substantial Pearson-correlation changes among numerical feature pairs.
def analyze_drift_relationships(
    reference,
    current,
    columns,
    minimum_change=0.20,
):
    """Return descriptive numerical correlation changes, not causal findings."""

    numerical_columns = [
        column
        for column in columns
        if (
            pd.api.types.is_numeric_dtype(reference[column])
            and pd.api.types.is_numeric_dtype(current[column])
        )
    ]

    output_columns = [
        "feature_a",
        "feature_b",
        "reference_correlation",
        "current_correlation",
        "change",
    ]

    if len(numerical_columns) < 2:
        return pd.DataFrame(columns=output_columns)

    reference_correlations = reference[numerical_columns].corr()
    current_correlations = current[numerical_columns].corr()
    relationships = []

    for index, feature_a in enumerate(numerical_columns):
        for feature_b in numerical_columns[index + 1:]:
            reference_value = reference_correlations.loc[feature_a, feature_b]
            current_value = current_correlations.loc[feature_a, feature_b]

            if (
                pd.notna(reference_value)
                and pd.notna(current_value)
                and abs(current_value - reference_value) >= minimum_change
            ):
                relationships.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "reference_correlation": round(
                            float(reference_value),
                            3,
                        ),
                        "current_correlation": round(
                            float(current_value),
                            3,
                        ),
                        "change": round(
                            float(current_value - reference_value),
                            3,
                        ),
                    }
                )

    if not relationships:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(relationships).sort_values(
        "change",
        key=lambda values: values.abs(),
        ascending=False,
        ignore_index=True,
    )


# Build a standalone HTML version of the existing analysis findings.
def build_html_report(results, relationships, alpha):
    """Build a self-contained HTML report for download."""

    significant_count = int(
        (results["status"] == "Significant drift").sum()
    )

    def table_rows(dataframe):
        """Return escaped HTML rows for a DataFrame."""

        return "".join(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in row
            )
            + "</tr>"
            for row in dataframe.itertuples(index=False, name=None)
        )

    feature_columns = [
        "feature",
        "feature_type",
        "status",
        "severity",
        "what_changed",
        "new_categories",
        "removed_categories",
    ]

    feature_rows = table_rows(results[feature_columns])
    relationship_rows = table_rows(relationships)

    if not relationship_rows:
        relationship_rows = (
            "<tr><td colspan='5'>No relationship changes met the "
            "reporting threshold.</td></tr>"
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>Data Drift Report</title>
<style>
body {{ font-family: Arial, sans-serif; color: #1f2933; margin: 40px; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #cbd5df; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #edf2f7; }}
h1, h2 {{ color: #102a43; }}
</style>
</head>
<body>
<h1>Data Drift Report</h1>
<p>Alpha: {alpha:.3f}. Significant drifted features: {significant_count} of {len(results)}.</p>
<p>Severity describes statistical drift magnitude, not business impact. Statistical drift does not by itself establish model failure or causation.</p>
<h2>Feature-Level Findings</h2>
<table>
<tr><th>Feature</th><th>Type</th><th>Status</th><th>Severity</th><th>What Changed</th><th>New Categories</th><th>Removed Categories</th></tr>
{feature_rows}
</table>
<h2>Changed Numerical Relationships</h2>
<p>Correlation changes describe association, not causation.</p>
<table>
<tr><th>Feature A</th><th>Feature B</th><th>Reference Correlation</th><th>Current Correlation</th><th>Change</th></tr>
{relationship_rows}
</table>
</body>
</html>"""
