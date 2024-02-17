from glob import glob

import pandas as pd
import streamlit as st


# Function to load data from a selected file
@st.cache_data
def load_data(file_name):
    return pd.read_json(file_name, lines=True)


def app():
    st.sidebar.title("MATH dataset viewer")

    # Sidebar to select the file name
    dataset_folder = ["data/math/*.jsonl"]
    files = [glob(path) for path in dataset_folder]
    files = sum(files, [])
    print(files)

    selected_file = st.sidebar.selectbox("Select file", files)
    if selected_file is None:
        st.write("No file selected.")
        return

    # Load data from the selected file
    df = load_data(selected_file)

    # Sidebar to select difficulty level
    level = st.sidebar.selectbox("Select difficulty level (MATH)", [1, 2, 3, 4, 5])

    # Filter dataframe based on selected difficulty level
    df_filtered = df[df["meta"].apply(lambda x: x.get("level", 0) == level)]

    if df_filtered.empty:
        st.write("No questions available for the selected difficulty level.")
    else:
        # Slider to select question number
        question_number = st.slider("Select a question number", min_value=0, max_value=df_filtered.shape[0] - 1)

        # Display question, answer, and metadata
        st.subheader("Question:")
        st.write(df_filtered.iloc[question_number]["question"])
        st.subheader("Answer:")
        st.write(df_filtered.iloc[question_number]["answer"])
        st.write("Meta data:", df_filtered.iloc[question_number]["meta"])


if __name__ == "__main__":
    app()
