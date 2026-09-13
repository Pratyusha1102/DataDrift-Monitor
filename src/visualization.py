# Create presentation-friendly charts for the DataDrift Monitor dashboard.

import pandas as pd


# Apply a consistent visual style to all dashboard charts.
def apply_chart_style(ax):

    # Make the chart background transparent so it blends with Streamlit.
    ax.set_facecolor("none")

    # Use light text so labels remain visible in the dark dashboard.
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    # Make tick labels readable on the dark interface.
    ax.tick_params(
        axis="both",
        colors="white",
        labelsize=9,
    )

    # Use subtle borders around the plotting area.
    for spine in ax.spines.values():
        spine.set_alpha(0.3)
        spine.set_color("white")

    # Use a subtle horizontal grid.
    ax.grid(
        axis="y",
        alpha=0.15,
    )


# Compare numerical feature distributions using overlaid histograms.
def plot_numerical_distribution(reference_series, current_series, feature_name):
    import matplotlib.pyplot as plt

    # Remove missing numerical values before plotting.
    reference_data = reference_series.dropna()
    current_data = current_series.dropna()

    # Choose a suitable number of bins based on the dataset size.
    total_values = len(reference_data) + len(current_data)

    if total_values <= 100:

        number_of_bins = 8

    elif total_values <= 500:

        number_of_bins = 12

    else:

        number_of_bins = 20

    # Create the figure and plotting area.
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot the reference distribution.
    ax.hist(
        reference_data,
        bins=number_of_bins,
        alpha=0.55,
        label="Reference",
    )

    # Plot the current distribution.
    ax.hist(
        current_data,
        bins=number_of_bins,
        alpha=0.55,
        label="Current",
    )

    # Add chart title and axis labels.
    ax.set_title(
        f"Distribution Comparison: {feature_name}",
        fontsize=14,
        pad=10,
    )

    ax.set_xlabel(
        feature_name,
        fontsize=10,
    )

    ax.set_ylabel(
        "Frequency",
        fontsize=10,
    )

    # Add the chart legend.
    ax.legend(
        frameon=False,
        fontsize=9,
    )

    # Improve readability of the chart.
    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    # Keep the chart background compatible with the dashboard.
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    # Make the chart fit its available space.
    fig.tight_layout(pad=1.2)

    # Return the completed figure to Streamlit.
    return fig


# Compare categorical feature proportions using grouped bars.
def plot_categorical_distribution(reference_series, current_series, feature_name):
    import matplotlib.pyplot as plt

    reference_data = reference_series.astype("object").fillna("[MISSING]")

    current_data = current_series.astype("object").fillna("[MISSING]")

    reference_proportions = reference_data.value_counts(normalize=True)

    current_proportions = current_data.value_counts(normalize=True)

    categories = reference_proportions.index.union(current_proportions.index)

    reference_proportions = reference_proportions.reindex(
        categories,
        fill_value=0,
    )

    current_proportions = current_proportions.reindex(
        categories,
        fill_value=0,
    )

    comparison = pd.DataFrame(
        {
            "Reference": reference_proportions,
            "Current": current_proportions,
        }
    )

    # Create a compact figure for the dashboard.
    fig, ax = plt.subplots(
        figsize=(10, 4),
    )

    fig.patch.set_alpha(0)

    comparison.plot(
        kind="bar",
        ax=ax,
        width=0.72,
        alpha=0.85,
    )

    ax.set_title(
        f"Category Distribution: {feature_name}",
        fontsize=14,
        pad=10,
    )

    ax.set_xlabel(
        feature_name,
        fontsize=10,
    )

    ax.set_ylabel(
        "Proportion",
        fontsize=10,
    )

    # Convert decimal proportions into percentage labels.
    tick_values = ax.get_yticks()

    ax.set_yticks(tick_values)

    ax.set_yticklabels([f"{value * 100:.0f}%" for value in tick_values])

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    # Keep category names horizontal for easier reading.
    ax.tick_params(
        axis="x",
        labelrotation=0,
    )

    apply_chart_style(ax)

    fig.tight_layout(pad=1.2)

    return fig


# Create a compact horizontal bar chart for overall drift status.
def plot_drift_summary(results_df):
    import matplotlib.pyplot as plt

    status_counts = (
        results_df["status"]
        .value_counts()
        .reindex(
            [
                "Significant drift",
                "Potential drift",
                "Needs review",
                "Stable",
            ],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0)

    # Draw the bars directly with Matplotlib instead of pandas plotting.
    values = status_counts.to_numpy()
    labels = status_counts.index.to_list()
    positions = range(len(labels))

    ax.barh(
        positions,
        values,
        height=0.65,
        alpha=0.85,
    )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)

    ax.set_title(
        "Drift Detection Summary",
        fontsize=14,
        pad=10,
    )

    ax.set_xlabel(
        "Number of Features",
        fontsize=10,
    )

    ax.set_ylabel("")

    # Keep status labels horizontal and readable.
    ax.tick_params(
        axis="y",
        labelrotation=0,
    )

    apply_chart_style(ax)

    # Add the feature count at the end of each bar.
    for index, value in enumerate(status_counts):

        ax.text(
            value + 0.05,
            index,
            str(value),
            va="center",
            fontsize=9,
            color="white",
        )

    fig.tight_layout(pad=1.2)

    return fig
