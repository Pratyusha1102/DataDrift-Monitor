import pandas as pd


# Check if the DataFrame is not empty.
def validate_not_empty(df):
    if df.empty:
        raise ValueError("The Dataset is empty.")
    return True


# Check if the DataFrame has columns.
def validate_has_columns(df):
    if len(df.columns) == 0:
        raise ValueError("The Dataset has no columns.")
    return True


# Check that the two datasets have at least one common column.
def validate_common_columns(reference_df, current_df):
    common_columns = set(reference_df.columns) & set(current_df.columns)

    if len(common_columns) == 0:
        raise ValueError("The datasets have no common columns.")
    return common_columns


# Check that common columns have compatible data types.
def validate_column_types(reference_df, current_df, common_columns):
    for column in common_columns:
        reference_is_numeric = pd.api.types.is_numeric_dtype(reference_df[column])
        current_is_numeric = pd.api.types.is_numeric_dtype(current_df[column])

        if reference_is_numeric != current_is_numeric:
            raise ValueError(
                f"Column '{column}' has incompatible data types "
                "between the datasets."
            )
    return True


# Return common columns in the order they appear in the reference dataset.
def get_common_columns(reference_df, current_df):
    return [column for column in reference_df.columns if column in current_df.columns]
