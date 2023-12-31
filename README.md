# Levenshtein Distance Matcher

## Description

This project is a text-matching tool that uses the Levenshtein distance algorithm to find the similarity between sets of strings. Built with Python and Streamlit, this tool allows users to upload CSV files or manually input strings for comparison. Users can also fine-tune the matching criteria using automatic or manual sliders.

![Screenshot](screenshot.png)

## Table of Contents

- [Levenshtein Distance Matcher](#levenshtein-distance-matcher)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [What is Levenshtein Distance?](#what-is-levenshtein-distance)
  - [Installation](#installation)
    - [Using pip](#using-pip)
  - [Usage](#usage)
    - [Upload CSV](#upload-csv)
    - [Manual Input](#manual-input)
    - [Filtering Method](#filtering-method)
    - [Case Sensitivity](#case-sensitivity)
    - [Download CSV](#download-csv)
  - [Contributing](#contributing)
  - [License](#license)
  - [Hosted WebApp](#hosted-webapp)

## What is Levenshtein Distance?

Learn more about Levenshtein distance on [Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)

## Installation

Clone the repository:

```bash
git clone https://github.com/dvonpasecky/fuzzy-matcher.git
cd fuzzy-matcher
```

### Using pip

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Upload CSV

1. Choose "Upload CSV" from the sidebar.
2. Upload your CSV file. The file should contain two columns of strings to be compared.

### Manual Input

1. Choose "Manual Input" from the sidebar.
2. Manually enter strings in the columns that appear.

### Filtering Method

Choose either the "Automatic Slider" or the "Manual Slider" to adjust the matching criteria.

### Case Sensitivity

Toggle the "Case Sensitive" option on for case-sensitive matching.

### Download CSV

Download the filtered results as a CSV file.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

## Hosted WebApp

You can access the hosted webapp [here](https://fuzzy-matcher.streamlit.app/).
