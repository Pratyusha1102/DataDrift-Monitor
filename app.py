# Build the user interface for the DataDrift Monitor dashboard.

from io import BytesIO

import streamlit as st

from src.data_loader import load_csv

from src.validator import (
    validate_not_empty,
    validate_has_columns,
    validate_common_columns,
    validate_column_types,
    get_common_columns,
)

from src.drift_detector import (
    detect_drift,
    get_feature_interpretation,
)

from src.insights import (
    add_phase_two_insights,
    analyze_drift_relationships,
    build_html_report,
    categorical_change_details,
)

from src.visualization import (
    plot_drift_summary,
    plot_numerical_distribution,
    plot_categorical_distribution,
)

# Load uploaded CSV bytes into a DataFrame and cache the result.
@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes):
    return load_csv(BytesIO(file_bytes))

# Run statistical drift analysis and cache the results for unchanged inputs.
@st.cache_data(show_spinner=False)
def run_drift_analysis(
    reference_df,
    current_df,
    common_columns,
    alpha,
):
    return detect_drift(
        reference_df,
        current_df,
        common_columns,
        alpha=alpha,
    )

# Configure the Streamlit page.
st.set_page_config(
    page_title="DataDrift Monitor",
    layout="wide",
)


# Add custom styling for the dashboard.
st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .section-description {
        font-size: 0.9rem;
        opacity: 0.75;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    .status-box {
        padding: 1rem;
        border-radius: 0.6rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Display the main dashboard heading.
st.markdown(
    '<div class="main-title">📊 DataDrift Monitor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Statistical Data Drift Detection Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)


# Configure analysis settings in the sidebar.
with st.sidebar:

    st.header("Analysis Settings")

    alpha = st.number_input(
        "Significance level (α)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.001,
        format="%.3f",
    )

    st.caption(
        f"Current significance threshold: α = {alpha:.3f}"
    )

    st.divider()

    with st.expander("How this dashboard works"):

        st.markdown(
            """
            **1. Upload two datasets**

            - Reference dataset: baseline or training-like data
            - Current dataset: newer data to compare

            **2. Numerical features**

            The two-sample Kolmogorov-Smirnov (KS) test
            compares their distributions.

            **3. Categorical features**

            The chi-square test compares category frequencies.
            Cramér's V reports the effect size.

            **4. Multiple testing**

            Bonferroni correction is applied to the p-values.

            **5. Interpretation**

            Results are classified as Stable, Potential drift,
            Significant drift, or Needs review.
            """
        )


# Create the file upload section.
st.subheader("Upload Datasets")

st.markdown(
    """
    <div class="section-description">
        Upload a reference dataset and a current dataset with
        matching feature names.
    </div>
    """,
    unsafe_allow_html=True,
)


# Place the two uploaders side by side.
upload_col1, upload_col2 = st.columns(2)


with upload_col1:

    reference_file = st.file_uploader(
        "Reference Dataset",
        type=["csv"],
        key="reference_file",
        help="Upload the baseline or training-like CSV dataset.",
    )


with upload_col2:

    current_file = st.file_uploader(
        "Current Dataset",
        type=["csv"],
        key="current_file",
        help="Upload the newer CSV dataset to check for drift.",
    )


# Add spacing before the analysis button.
st.write("")


# Center the Analyze button.
button_col1, button_col2, button_col3 = st.columns(
    [1, 1, 1]
)


with button_col2:

    analyze_button = st.button(
        "Analyze Drift",
        type="primary",
        width="stretch",
    )


# Run the analysis when the user clicks the button.
if analyze_button:

    # Make sure both datasets have been uploaded.
    if reference_file is None or current_file is None:

        st.error(
            "Please upload both a reference dataset and a current dataset."
        )

    else:

        try:

            # Show a progress indicator while the datasets are processed.
            with st.spinner("Loading and analyzing datasets..."):

                # Load both CSV files.
                reference_df = load_uploaded_csv(reference_file.getvalue())
                current_df = load_uploaded_csv(current_file.getvalue())

                # Validate that both datasets contain data.
                validate_not_empty(reference_df)
                validate_not_empty(current_df)

                # Validate that both datasets contain columns.
                validate_has_columns(reference_df)
                validate_has_columns(current_df)

                # Validate that at least one feature exists in both datasets.
                validate_common_columns(
                    reference_df,
                    current_df,
                )

                # Get common columns in reference dataset order.
                common_columns = get_common_columns(
                    reference_df,
                    current_df,
                )

                # Validate that common columns have compatible types.
                validate_column_types(
                    reference_df,
                    current_df,
                    common_columns,
                )

                # Run the cached statistical drift analysis while showing processing feedback.
                with st.spinner("Analyzing datasets for statistical drift..."):
                    results_df = run_drift_analysis(
                        reference_df,
                        current_df,
                        common_columns,
                        alpha,
                    )

                    # Add descriptive findings without changing test results.
                    results_df = add_phase_two_insights(
                        results_df,
                        reference_df,
                        current_df,
                    )

                    # Calculate numerical association changes separately.
                    relationships_df = analyze_drift_relationships(
                        reference_df,
                        current_df,
                        common_columns,
                    )

            # Store the analysis results in Streamlit session state.
            st.session_state["reference_df"] = reference_df
            st.session_state["current_df"] = current_df
            st.session_state["results_df"] = results_df
            st.session_state["common_columns"] = common_columns
            st.session_state["alpha"] = alpha
            st.session_state["relationships_df"] = relationships_df

            st.success(
                "Drift analysis completed successfully."
            )

        except Exception as error:

            # Display validation or processing errors clearly.
            st.error(
                f"Analysis failed: {error}"
            )


# Display the dashboard after successful analysis.
if "results_df" in st.session_state:

    # Retrieve previously calculated results.
    reference_df = st.session_state["reference_df"]
    current_df = st.session_state["current_df"]
    results_df = st.session_state["results_df"]
    common_columns = st.session_state["common_columns"]
    alpha = st.session_state["alpha"]
    relationships_df = st.session_state["relationships_df"]

    st.divider()

    # ---------------------------------------------------------
    # DATASET SUMMARY
    # ---------------------------------------------------------

    st.subheader("Dataset Summary")

    st.markdown(
        """
        <div class="section-description">
            Basic information about the two datasets used in the comparison.
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:

        st.metric(
            "Reference Rows",
            f"{len(reference_df):,}",
        )

    with summary_col2:

        st.metric(
            "Current Rows",
            f"{len(current_df):,}",
        )

    with summary_col3:

        st.metric(
            "Common Features",
            len(common_columns),
        )

    with summary_col4:

        st.metric(
            "Reference Columns",
            len(reference_df.columns),
        )


    st.write("")


    # ---------------------------------------------------------
    # DRIFT SUMMARY
    # ---------------------------------------------------------

    st.subheader("Drift Summary")

    st.markdown(
        """
        <div class="section-description">
            Overall feature-level results after Bonferroni correction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Count each drift status.
    significant_count = int(
        (results_df["status"] == "Significant drift").sum()
    )

    potential_count = int(
        (results_df["status"] == "Potential drift").sum()
    )

    needs_review_count = int(
        (results_df["status"] == "Needs review").sum()
    )

    stable_count = int(
        (results_df["status"] == "Stable").sum()
    )


    # Display summary metrics.
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        st.metric(
            "Significant Drift",
            significant_count,
        )

    with metric_col2:

        st.metric(
            "Potential Drift",
            potential_count,
        )

    with metric_col3:

        st.metric(
            "Needs Review",
            needs_review_count,
        )

    with metric_col4:

        st.metric(
            "Stable",
            stable_count,
        )


    st.write("")


    # Determine the overall dashboard message.
    if significant_count > 0:

        st.warning(
            f"{significant_count} feature(s) show statistically "
            "significant drift after Bonferroni correction."
        )

    elif potential_count > 0:

        st.info(
            f"{potential_count} feature(s) show potential drift, "
            "but do not remain significant after multiple-testing correction."
        )

    elif needs_review_count > 0:

        st.warning(
            f"{needs_review_count} feature(s) require review because "
            "the statistical assumptions may be unreliable."
        )

    else:

        st.success(
            "No statistically significant feature-level drift was detected."
        )


    # Display the compact drift summary chart.
    with st.container(border=True):

        summary_chart = plot_drift_summary(
            results_df
        )

        st.pyplot(
            summary_chart,
            width="stretch",
        )


    # ---------------------------------------------------------
    # DRIFT SEVERITY AND INVESTIGATION
    # ---------------------------------------------------------

    st.subheader("Drift Severity and Investigation")

    st.markdown(
        """
        <div class="section-description">
            Severity describes the magnitude of the statistical distribution
            difference. It is not a business-impact or model-risk score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        results_df[
            [
                "feature",
                "feature_type",
                "status",
                "severity",
                "what_changed",
            ]
        ].rename(
            columns={
                "feature": "Feature",
                "feature_type": "Type",
                "status": "Status",
                "severity": "Severity",
                "what_changed": "What Changed",
            }
        ),
        width="stretch",
        hide_index=True,
    )


    # ---------------------------------------------------------
    # FEATURE RESULTS TABLE
    # ---------------------------------------------------------

    st.subheader("Feature-Level Results")

    st.markdown(
        """
        <div class="section-description">
            Statistical test results for every common feature.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Create a presentation-friendly copy of the results table.
    display_df = results_df.copy()

    display_df = display_df.rename(
        columns={
            "feature": "Feature",
            "feature_type": "Type",
            "statistic": "Test Statistic",
            "p_value": "Raw p-value",
            "adjusted_p_value": "Adjusted p-value",
            "effect_size": "Cramér's V",
            "reference_missing_pct": "Reference Missing %",
            "current_missing_pct": "Current Missing %",
            "needs_review": "Review",
            "status": "Status",
            "severity": "Severity",
            "what_changed": "What Changed",
        }
    )


    # Display the results table.
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Test Statistic": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Raw p-value": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "Adjusted p-value": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "Cramér's V": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Reference Missing %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Current Missing %": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )


    # Explain the four possible statuses.
    with st.expander("How to interpret the drift statuses"):

        st.markdown(
            f"""
            **Stable**

            Adjusted p-value ≥ {alpha:.3f}. There is no statistically
            significant evidence of distribution change.

            **Potential drift**

            Raw p-value < {alpha:.3f}, but the Bonferroni-adjusted
            p-value is not below the significance threshold.

            **Significant drift**

            Adjusted p-value < {alpha:.3f}. The feature shows
            statistically significant distribution change after
            multiple-testing correction.

            **Needs review**

            The statistical comparison has an assumption or data-quality
            issue, such as sparse expected categorical frequencies.
            """
        )


    # ---------------------------------------------------------
    # DOWNLOAD RESULTS
    # ---------------------------------------------------------

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(
            label="Download Full Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="data_drift_results.csv",
            mime="text/csv",
            use_container_width=False,
        )

    with download_col2:

        st.download_button(
            label="Download HTML Report",
            data=build_html_report(
                results_df,
                relationships_df,
                alpha,
            ),
            file_name="data_drift_report.html",
            mime="text/html",
            use_container_width=False,
        )


    # ---------------------------------------------------------
    # DRIFT RELATIONSHIPS
    # ---------------------------------------------------------

    if not relationships_df.empty:

        st.subheader("Drift Relationships")

        st.markdown(
            """
            <div class="section-description">
                Pearson-correlation changes among numerical features.
                These describe association, not causation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            relationships_df.rename(
                columns={
                    "feature_a": "Feature A",
                    "feature_b": "Feature B",
                    "reference_correlation": "Reference Correlation",
                    "current_correlation": "Current Correlation",
                    "change": "Correlation Change",
                }
            ),
            width="stretch",
            hide_index=True,
        )


    st.divider()


    # ---------------------------------------------------------
    # FEATURE-LEVEL ANALYSIS
    # ---------------------------------------------------------

    st.subheader("Feature-Level Analysis")

    st.markdown(
        """
        <div class="section-description">
            Select an individual feature to inspect its statistical
            result and distribution comparison.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Allow the user to select one feature for detailed analysis.
    selected_feature = st.selectbox(
        "Select a feature",
        common_columns,
    )


    # Retrieve the result row for the selected feature.
    selected_result = results_df[
        results_df["feature"] == selected_feature
    ].iloc[0]


    # Display the feature's current status.
    status = selected_result["status"]


    if status == "Significant drift":

        st.error(
            "Status: Significant drift"
        )

    elif status == "Potential drift":

        st.warning(
            "Status: Potential drift"
        )

    elif status == "Needs review":

        st.warning(
            "Status: Needs review"
        )

    else:

        st.success(
            "Status: Stable"
        )


    # Display the main statistical values.
    detail_col1, detail_col2, detail_col3 = st.columns(3)


    with detail_col1:

        st.metric(
            "Test Statistic",
            f"{selected_result['statistic']:.4f}",
        )


    with detail_col2:

        st.metric(
            "Raw p-value",
            f"{selected_result['p_value']:.6f}",
        )


    with detail_col3:

        st.metric(
            "Adjusted p-value",
            f"{selected_result['adjusted_p_value']:.6f}",
        )


    # Display feature type and effect size information.
    info_col1, info_col2 = st.columns(2)


    with info_col1:

        st.markdown(
            f"**Feature Type:** {selected_result['feature_type'].title()}"
        )


    with info_col2:

        if selected_result["feature_type"] == "categorical":

            st.markdown(
                f"**Cramér's V:** "
                f"{selected_result['effect_size']:.4f}"
            )

        else:

            st.markdown(
                "**Effect Size:** Not applicable for numerical features."
            )


    st.write("")


    # ---------------------------------------------------------
    # MISSING VALUE INFORMATION
    # ---------------------------------------------------------

    missing_col1, missing_col2 = st.columns(2)


    with missing_col1:

        st.metric(
            "Reference Missing",
            f"{selected_result['reference_missing_pct']:.2f}%",
        )


    with missing_col2:

        st.metric(
            "Current Missing",
            f"{selected_result['current_missing_pct']:.2f}%",
        )


    st.write("")


    # ---------------------------------------------------------
    # NUMERICAL FEATURE DETAILS
    # ---------------------------------------------------------

    if selected_result["feature_type"] == "numerical":

        reference_values = (
            reference_df[selected_feature]
            .dropna()
        )

        current_values = (
            current_df[selected_feature]
            .dropna()
        )


        # Calculate descriptive statistics.
        reference_mean = reference_values.mean()
        current_mean = current_values.mean()

        reference_median = reference_values.median()
        current_median = current_values.median()


        st.markdown("**Descriptive Statistics**")


        stats_col1, stats_col2 = st.columns(2)


        with stats_col1:

            st.markdown("**Reference Dataset**")

            st.write(
                f"Mean: `{reference_mean:.4f}`"
            )

            st.write(
                f"Median: `{reference_median:.4f}`"
            )


        with stats_col2:

            st.markdown("**Current Dataset**")

            st.write(
                f"Mean: `{current_mean:.4f}`"
            )

            st.write(
                f"Median: `{current_median:.4f}`"
            )


    # ---------------------------------------------------------
    # FEATURE INTERPRETATION
    # ---------------------------------------------------------

    st.write("")


    interpretation = get_feature_interpretation(
        selected_result
    )


    st.info(
        interpretation
    )


    # ---------------------------------------------------------
    # WHAT CHANGED
    # ---------------------------------------------------------

    st.markdown("**What Changed?**")

    st.write(
        selected_result["what_changed"]
    )

    if selected_result["feature_type"] == "categorical":

        added_categories, removed_categories, category_changes = (
            categorical_change_details(
                reference_df[selected_feature],
                current_df[selected_feature],
            )
        )

        category_col1, category_col2 = st.columns(2)

        with category_col1:

            st.markdown(
                "**New Categories:** "
                + (
                    ", ".join(added_categories)
                    if added_categories
                    else "None"
                )
            )

        with category_col2:

            st.markdown(
                "**Removed Categories:** "
                + (
                    ", ".join(removed_categories)
                    if removed_categories
                    else "None"
                )
            )

        st.dataframe(
            category_changes,
            width="stretch",
            hide_index=True,
        )


    # ---------------------------------------------------------
    # DISTRIBUTION VISUALIZATION
    # ---------------------------------------------------------

    st.markdown("**Distribution Comparison**")


    with st.container(border=True):

        if selected_result["feature_type"] == "numerical":

            distribution_chart = plot_numerical_distribution(
                reference_df[selected_feature],
                current_df[selected_feature],
                selected_feature,
            )

        else:

            distribution_chart = plot_categorical_distribution(
                reference_df[selected_feature],
                current_df[selected_feature],
                selected_feature,
            )

        st.pyplot(
            distribution_chart,
            width="stretch",
        )


    # ---------------------------------------------------------
    # STATISTICAL METHODOLOGY
    # ---------------------------------------------------------

    with st.expander("Statistical Methodology"):

        st.markdown(
            """
            ### Numerical Features — KS Test

            The two-sample Kolmogorov-Smirnov test compares the
            distributions of a numerical feature in the reference
            and current datasets.

            **Null hypothesis (H₀):**

            The reference and current distributions are the same.

            **Alternative hypothesis (H₁):**

            The reference and current distributions are different.


            ### Categorical Features — Chi-Square Test

            The chi-square test compares category frequencies between
            the reference and current datasets.

            **Null hypothesis (H₀):**

            Category proportions are independent of dataset group.

            **Alternative hypothesis (H₁):**

            Category proportions differ between the two datasets.


            ### Cramér's V

            Cramér's V is used as an effect-size measure for
            categorical distribution differences.


            ### Bonferroni Correction

            Because multiple features are tested simultaneously,
            Bonferroni correction is applied.

            The adjusted significance threshold is:

            **α / m**

            where:

            - α is the selected significance level
            - m is the number of tested features


            ### Missing Values

            Missing numerical values are excluded from the KS test
            calculation and their percentage is reported separately.

            Missing categorical values are treated as an explicit
            `[MISSING]` category.


            ### Important Note

            A statistically significant result indicates evidence
            that the feature distribution changed. It does not,
            by itself, establish why the change occurred or whether
            the change is harmful to a machine-learning model.
            """
        )
