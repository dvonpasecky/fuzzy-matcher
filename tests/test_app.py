"""Unit tests for fuzzy-matcher app functions.

This module tests:
- State initialization
- Levenshtein calculation
- Filtering by slider
- File upload handling
"""

import pandas as pd
import pytest

from app import (
    DefaultSliderValues,
    DefaultStrings,
    calculate_levenshtein,
    filter_by_slider,
    handle_file_upload,
    initialize_state,
)


@pytest.fixture
def sample_df():
    """Fixture providing a sample DataFrame for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample string pairs and Levenshtein distances.

    """
    return pd.DataFrame(
        {
            "String1": ["apple", "orange", "banana"],
            "String2": ["apple", "orange", "mango"],
            "LevenshteinDistance": [0, 0, 3],
        }
    )


def test_initialize_state(mocker):
    """Test the initialization of Streamlit session state variables."""
    mock_session_state = mocker.MagicMock()
    mocker.patch("app.st.session_state", mock_session_state)

    initialize_state()

    mock_session_state.get.assert_any_call(
        "auto_slider_value", DefaultSliderValues.AUTO.value
    )
    mock_session_state.get.assert_any_call(
        "manual_slider_value", DefaultSliderValues.MANUAL.value
    )
    mock_session_state.get.assert_any_call("num_strings", DefaultStrings.NUM.value)


def test_calculate_levenshtein_structure():
    """Test the structure of the DataFrame returned by calculate_levenshtein."""
    column1 = ["apple", "orange"]
    column2 = ["apple", "orange"]
    case_sensitive = True
    df = calculate_levenshtein(column1, column2, case_sensitive)
    assert not df.empty
    assert "String1" in df.columns
    assert "String2" in df.columns
    assert "LevenshteinDistance" in df.columns
    assert df.iloc[0]["LevenshteinDistance"] == 0


def test_calculate_levenshtein_values():
    """Test the values in the DataFrame returned by calculate_levenshtein."""
    column1 = ["apple"]
    column2 = ["apple", "aplpe"]
    df = calculate_levenshtein(column1, column2)
    row = df[(df["String1"] == "apple") & (df["String2"] == "apple")]
    assert row["LevenshteinDistance"].iloc[0] == 0


def test_filter_by_slider_with_sample_data(sample_df):
    """Test filtering by slider using sample data."""
    threshold = 2
    filtered_df = filter_by_slider(sample_df, threshold)

    assert not filtered_df.empty
    assert filtered_df["LevenshteinDistance"].max() <= threshold


def test_filter_by_slider_with_known_data():
    """Test filtering by slider with known data."""
    df = pd.DataFrame(
        {
            "String1": ["apple", "apple"],
            "String2": ["apple", "aplpe"],
            "LevenshteinDistance": [0, 2],
        }
    )
    filtered_df = filter_by_slider(df, 1)
    assert len(filtered_df) == 1
    assert filtered_df.loc[1, "LevenshteinDistance"] == 0


def test_case_insensitivity():
    """Test case insensitivity in Levenshtein calculation."""
    column1 = ["Apple", "Orange"]
    column2 = ["apple", "orange"]
    case_sensitive = False

    df = calculate_levenshtein(column1, column2, case_sensitive)
    assert df.iloc[0]["LevenshteinDistance"] == 0


def test_empty_input():
    """Test handling of empty input lists."""
    df = calculate_levenshtein([], [], True)
    assert df.empty


def test_non_string_input():
    """Test handling of non-string input."""
    df = calculate_levenshtein([str(x) for x in [1, 2, 3]], ["1", "2", "3"], True)
    assert not df.empty
    assert df.iloc[0]["LevenshteinDistance"] == 0


def test_case_sensitivity():
    """Test case sensitivity in Levenshtein calculation."""
    column1 = ["Apple", "Orange"]
    column2 = ["apple", "orange"]
    case_sensitive = True

    df = calculate_levenshtein(column1, column2, case_sensitive)
    assert df.iloc[0]["LevenshteinDistance"] != 0


def test_large_levenshtein_distance():
    """Test calculation of large Levenshtein distances."""
    column1 = ["apple"]
    column2 = ["orange"]
    df = calculate_levenshtein(column1, column2, True)
    assert df.iloc[0]["LevenshteinDistance"] > 0


def test_zero_threshold(sample_df):
    """Test filtering with zero threshold."""
    filtered_df = filter_by_slider(sample_df, 0)
    assert not filtered_df.empty
    assert filtered_df["LevenshteinDistance"].max() == 0


def test_large_threshold(sample_df):
    """Test filtering with large threshold."""
    filtered_df = filter_by_slider(sample_df, 100)
    assert not filtered_df.empty
    assert filtered_df["LevenshteinDistance"].max() <= 100


def test_handle_file_upload(mocker):
    """Test file upload handling."""
    mocker.patch("app.st.sidebar.file_uploader", return_value=None)
    assert handle_file_upload() == ([], [])
