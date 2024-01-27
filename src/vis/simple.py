import streamlit as st

import pandas as pd


@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_json("data/math-test-subset.jsonl", lines=True)


def app():
    df = load_data()

    question_number = st.slider("Select a question number", min_value=0, max_value=df.shape[0] - 1)

    st.write("Question:", df.loc[question_number, "question"])
    st.write("Answer:", df.loc[question_number, "answer"])
    st.write("Meta data:", df.loc[question_number, "meta"])


if __name__ == "__main__":
    app()
