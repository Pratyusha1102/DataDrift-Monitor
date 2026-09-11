# Test the dataset validation functions used by the drift detector.

import pandas as pd
import pytest

from src.validator import (
    validate_not_empty,
    validate_has_columns,
    validate_common_columns,
    validate_column_types,
    get_common_columns,
)


# Verify that a normal DataFrame passes the non-empty validation.
def test_validate_not_empty():
    df = pd.DataFrame({"age": [20, 21, 22]})

    assert validate_not_empty(df) is True


# Verify that an empty DataFrame is rejected.
def test_validate_not_empty_rejects_empty_dataset():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_not_empty(df)


# Verify that a DataFrame containing columns passes validation.
def test_validate_has_columns():
    df = pd.DataFrame({"age": [20, 21, 22]})

    assert validate_has_columns(df) is True


# Verify that two datasets with a shared column pass common-column validation.
def test_validate_common_columns():
    reference_df = pd.DataFrame({"age": [20, 21], "city": ["A", "B"]})
    current_df = pd.DataFrame({"age": [22, 23], "city": ["B", "C"]})

    common_columns = validate_common_columns(reference_df, current_df)

    assert common_columns == {"age", "city"}


# Verify that datasets without shared columns are rejected.
def test_validate_common_columns_rejects_no_common_columns():
    reference_df = pd.DataFrame({"age": [20, 21]})
    current_df = pd.DataFrame({"income": [100, 200]})

    with pytest.raises(ValueError):
        validate_common_columns(reference_df, current_df)


# Verify that compatible column types pass validation.
def test_validate_column_types():
    reference_df = pd.DataFrame({"age": [20, 21]})
    current_df = pd.DataFrame({"age": [22, 23]})

    assert validate_column_types(reference_df, current_df, {"age"}) is True


# Verify that numeric versus categorical type mismatches are rejected.
def test_validate_column_types_rejects_mismatch():
    reference_df = pd.DataFrame({"age": [20, 21]})
    current_df = pd.DataFrame({"age": ["22", "23"]})

    with pytest.raises(ValueError):
        validate_column_types(reference_df, current_df, {"age"})


# Verify that common columns retain the reference dataset's column order.
def test_get_common_columns_preserves_reference_order():
    reference_df = pd.DataFrame(
        {
            "age": [20, 21],
            "education": ["A", "B"],
            "income": [100, 200],
        }
    )

    current_df = pd.DataFrame(
        {
            "income": [300, 400],
            "age": [22, 23],
            "education": ["B", "C"],
        }
    )

    result = get_common_columns(reference_df, current_df)

    assert result == [
        "age",
        "education",
        "income",
    ]
