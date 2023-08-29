import itertools
from typing import List, Tuple

import pandas as pd
import streamlit as st
from Levenshtein import distance as levenshtein_distance

MAX_INT_32 = 2**31 - 1  # Maximum 32-bit integer


def initialize_state():
    """Initialize Streamlit session state variables."""
    st.session_state.auto_slider_value = st.session_state.get("auto_slider_value", 80)
    st.session_state.manual_slider_value = st.session_state.get(
        "manual_slider_value", 1
    )
    st.session_state.num_strings = st.session_state.get("num_strings", 1)


def handle_file_upload() -> Tuple[List[str], List[str]]:
    """Handles file upload and returns columns as lists."""
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).astype(str).fillna("")
        return df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()

    return [], []


def handle_manual_input() -> Tuple[List[str], List[str]]:
    """Handles manual input and returns columns as lists."""
    if "num_strings" not in st.session_state:
        st.session_state.num_strings = 1

    col_button1, col_button2 = st.sidebar.columns(2)

    if col_button1.button("Add a Row"):
        st.session_state.num_strings += 1

    if col_button2.button("Remove a Row"):
        st.session_state.num_strings = max(1, st.session_state.num_strings - 1)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        column1 = [
            st.text_input(f"Column 1, String {i+1}") or ""
            for i in range(st.session_state.num_strings)
        ]
    with col2:
        column2 = [
            st.text_input(f"Column 2, String {i+1}") or ""
            for i in range(st.session_state.num_strings)
        ]

    return column1, column2


def calculate_levenshtein(
    column1: List[str], column2: List[str], case_sensitive: bool = True
) -> pd.DataFrame:
    """Calculates Levenshtein distance between two lists of strings."""
    if not case_sensitive:
        column1 = [x.lower() for x in column1]
        column2 = [y.lower() for y in column2]

    column1 = list(set(column1))
    column2 = list(set(column2))

    data = [
        {
            "String1": str1,
            "String2": str2,
            "LevenshteinDistance": levenshtein_distance(str1, str2),
        }
        for str1, str2 in itertools.product(column1, column2)
    ]

    return pd.DataFrame(data)


def filter_by_slider(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Filters a DataFrame based on a Levenshtein distance threshold."""
    filtered_df = df[df["LevenshteinDistance"] <= threshold].copy()
    filtered_df.sort_values(by="LevenshteinDistance", inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.index += 1
    return filtered_df


def handle_automatic_slider(max_distance: int) -> int:
    """Handles automatic slider value and returns the calculated threshold."""
    st.latex(
        r"\text{Threshold} = \text{Max Distance} \times \left(1 - \frac{\text{Slider Value}}{100}\right)"
    )
    slider_value = st.slider(
        "Automatic Levenshtein Distance Filter",
        min_value=0,
        max_value=100,
        value=st.session_state.auto_slider_value,
    )
    st.session_state.auto_slider_value = slider_value
    threshold = (
        int(max_distance * (1 - (slider_value / 100))) if max_distance != 0 else 0
    )
    st.latex(
        f"\\text{{{threshold}}} = {max_distance} \\times \\left(1 - \\frac{{{slider_value}}}{{100}}\\right)"
    )
    return threshold


def handle_manual_slider(max_distance: int) -> int:
    st.latex(
        r"\text{Automatic Slider Value} = 100 \times \left(1 - \frac{\text{Manual Threshold}}{\text{Max Distance}}\right)"
    )
    slider_value = st.slider(
        "Manual Levenshtein Distance Threshold",
        min_value=0,
        max_value=max(max_distance, 1),
        value=st.session_state.manual_slider_value,
    )
    st.session_state.manual_slider_value = slider_value

    if max_distance != 0:
        st.session_state.auto_slider_value = int(
            100 * (1 - (slider_value / max_distance))
        )
    else:
        st.session_state.auto_slider_value = 0

    st.latex(
        f"\\text{{{st.session_state.auto_slider_value}}} = 100 \\times \\left(1 - \\frac{{{slider_value}}}{{{max_distance}}}\\right)"
    )

    return 0 if st.session_state.auto_slider_value == 100 else slider_value


def main():
    """Main function to run the Streamlit app."""
    st.title("Levenshtein Distance Matcher")
    st.markdown(
        "Learn more about Levenshtein distance on [Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)"
    )
    st.sidebar.header("User Options")
    initialize_state()

    input_method = st.sidebar.radio(
        "Choose input method", ["Upload CSV File", "Manual Input"]
    )
    column1, column2 = (
        handle_file_upload()
        if input_method == "Upload CSV File"
        else handle_manual_input()
    )
    if not column1 or not column2:
        return

    filter_method = st.sidebar.radio(
        "Choose filtering method", ["Automatic Slider", "Manual Slider"]
    )
    case_sensitive = st.sidebar.toggle(
        "Case Sensitive", value=True, help="Toggle on for case sensitivity"
    )

    df = calculate_levenshtein(column1, column2, case_sensitive)
    max_distance = 1 if df.empty else df["LevenshteinDistance"].max()

    threshold = (
        handle_automatic_slider(max_distance)
        if filter_method == "Automatic Slider"
        else handle_manual_slider(max_distance)
    )
    threshold = min(threshold, MAX_INT_32)

    filtered_df = filter_by_slider(df, threshold)
    st.dataframe(filtered_df)


if __name__ == "__main__":
    main()
