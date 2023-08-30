import itertools
from enum import Enum
from io import BytesIO
from random import shuffle
from typing import List, Tuple

import pandas as pd
import streamlit as st
from Levenshtein import distance as levenshtein_distance

MAX_INT_32 = 2**31 - 1  # Maximum 32-bit integer


class DefaultSliderValues(Enum):
    AUTO = 80
    MANUAL = 1


class DefaultStrings(Enum):
    NUM = 1


def initialize_state():
    """
    Initialize Streamlit session state variables.
    """
    st.session_state.auto_slider_value = st.session_state.get(
        "auto_slider_value", DefaultSliderValues.AUTO.value
    )
    st.session_state.manual_slider_value = st.session_state.get(
        "manual_slider_value", DefaultSliderValues.MANUAL.value
    )
    st.session_state.num_strings = st.session_state.get(
        "num_strings", DefaultStrings.NUM.value
    )


def handle_file_upload() -> Tuple[List[str], List[str]]:
    """Handle file upload and return columns as lists.

    Returns:
        Tuple[List[str], List[str]]: Two lists containing data from the uploaded CSV file,
        one for each column.
    """
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).astype(str).fillna("")
        return df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()

    return [], []


def get_input_fields(num_strings: int) -> Tuple[List[str], List[str]]:
    """Get input fields for manual input.

    Args:
        num_strings (int): Number of input fields for each column.

    Returns:
        Tuple[List[str], List[str]]: Two lists containing the input from the user,
        one for each column.
    """
    col1, col2 = st.sidebar.columns(2)
    column1 = []
    column2 = []

    with col1:
        for i in range(num_strings):
            default_val = st.session_state.get(f"col1_str_{i+1}", "")
            column1.append(st.text_input(f"Column 1, String {i+1}", default_val))

    with col2:
        for i in range(num_strings):
            default_val = st.session_state.get(f"col2_str_{i+1}", "")
            column2.append(st.text_input(f"Column 2, String {i+1}", default_val))

    return column1, column2


def handle_manual_input() -> Tuple[List[str], List[str]]:
    """Handle manual input and return columns as lists.

    Returns:
        Tuple[List[str], List[str]]: Two lists containing input from the user,
        one for each column.
    """
    if "num_strings" not in st.session_state:
        st.session_state.num_strings = 1

    col_button1, col_button2 = st.sidebar.columns(2)

    if col_button1.button("Add a Row"):
        st.session_state.num_strings += 1

    if col_button2.button("Remove a Row"):
        st.session_state.num_strings = max(1, st.session_state.num_strings - 1)

    return get_input_fields(st.session_state.num_strings)


def calculate_levenshtein(
    column1: List[str], column2: List[str], case_sensitive: bool = True
) -> pd.DataFrame:
    """Calculate Levenshtein distance between two lists of strings.

    Args:
        column1 (List[str]): The first list of strings.
        column2 (List[str]): The second list of strings.
        case_sensitive (bool, optional): Whether or not the calculation should be case-sensitive. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated Levenshtein distances.
    """
    column1 = [str(x) for x in column1]
    column2 = [str(y) for y in column2]

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
    """Filter a DataFrame based on a Levenshtein distance threshold.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        threshold (int): The maximum Levenshtein distance for a record to be included.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    filtered_df = df[df["LevenshteinDistance"] <= threshold].copy()
    filtered_df.sort_values(by="LevenshteinDistance", inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.index += 1
    return filtered_df


def handle_automatic_slider(max_distance: int) -> int:
    """Handle automatic slider value and return the calculated threshold.

    Args:
        max_distance (int): The maximum Levenshtein distance in the current dataset.

    Returns:
        int: The calculated threshold based on the automatic slider value.
    """
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
    """Handle manual slider value and return the threshold.

    Args:
        max_distance (int): The maximum Levenshtein distance in the current dataset.

    Returns:
        int: The threshold set by the manual slider.
    """
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


def create_download_button(
    df: pd.DataFrame, filename: str = "filtered_data.csv"
) -> None:
    """Create a Streamlit button for downloading a DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be downloaded.
        filename (str): The name of the downloaded file.
    """
    if df.empty:
        return

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button(
        "Download CSV",
        csv_buffer,
        file_name=filename,
        mime="text/csv",
    )


def generate_demo_data() -> Tuple[List[str], List[str]]:
    """
    Generate demo data with specified Levenshtein distances.

    Returns:
        Tuple[List[str], List[str]]: Two lists containing strings for demo.
    """

    column1 = ["apple", "banana", "pear", "kiwi", "carrot"]
    column2 = ["apple", "BANANA", "pears", "kiwiz", "carrotxx"]

    shuffle(column1)
    shuffle(column2)

    return column1, column2


def main():
    """Main function to run the Streamlit app.

    The main function orchestrates the entire Streamlit application, from user input
    to displaying results. It uses the helper functions defined in the module.
    """
    st.title("Levenshtein Distance Matcher")
    st.markdown(
        "Learn more about Levenshtein distance on [Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)"
    )
    st.sidebar.header("User Options")
    initialize_state()

    if "input_method" not in st.session_state:
        st.session_state.input_method = "Upload CSV"

    if st.sidebar.button("Generate Demo Data"):
        column1, column2 = generate_demo_data()
        st.session_state.num_strings = len(column1)

        # Update session state with demo data and switch to manual input
        for i, (val1, val2) in enumerate(zip(column1, column2)):
            st.session_state[f"col1_str_{i+1}"] = val1
            st.session_state[f"col2_str_{i+1}"] = val2
        st.session_state.input_method = "Manual Input"

    input_method = st.sidebar.radio(
        "Choose input method",
        ["Upload CSV", "Manual Input"],
        index=0 if st.session_state.input_method == "Upload CSV" else 1,
    )
    st.session_state.input_method = input_method

    column1, column2 = (
        handle_file_upload() if input_method == "Upload CSV" else handle_manual_input()
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

    if not filtered_df.empty:
        create_download_button(filtered_df)


if __name__ == "__main__":
    main()
