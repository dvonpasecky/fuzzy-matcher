import pytest
import pandas as pd
from app import calculate_levenshtein, filter_by_slider


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "String1": ["apple", "orange", "banana"],
            "String2": ["apple", "orange", "mango"],
            "LevenshteinDistance": [0, 0, 3],
        }
    )


def test_calculate_levenshtein():
    column1 = ["apple", "orange"]
    column2 = ["apple", "orange"]
    case_sensitive = True

    df = calculate_levenshtein(column1, column2, case_sensitive)

    assert not df.empty
    assert "String1" in df.columns
    assert "String2" in df.columns
    assert "LevenshteinDistance" in df.columns
    assert df.iloc[0]["LevenshteinDistance"] == 0


def test_filter_by_slider(sample_df):
    threshold = 2
    filtered_df = filter_by_slider(sample_df, threshold)

    assert not filtered_df.empty
    assert filtered_df["LevenshteinDistance"].max() <= threshold


def test_case_insensitivity():
    column1 = ["Apple", "Orange"]
    column2 = ["apple", "orange"]
    case_sensitive = False

    df = calculate_levenshtein(column1, column2, case_sensitive)
    assert df.iloc[0]["LevenshteinDistance"] == 0
