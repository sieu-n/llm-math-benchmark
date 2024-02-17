import asyncio
import json
from collections import Counter
from glob import glob
from pathlib import Path

import streamlit as st

from src.hendrycks import parse_prediction
from src.openai_assistant import parse_asssistant_steps
from src.openai_utils import syncclient


@st.cache_data(persist=True)
def load_json(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


@st.cache_data(persist=True)
def retrieve_run(thread_id, run_id):
    resp = syncclient.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return resp


def load_math_dataset_ui():
    folders = glob("./results/math-csv-repro/exp-*/")
    exp_name = st.sidebar.selectbox("Select experiment name", folders)
    exp_name = Path(exp_name)
    return {
        "exp_name": exp_name,
        "raw": load_json(exp_name / "output-raw.json"),
        "str": load_json(exp_name / "output-str.json"),
        "predictions": load_json(exp_name / "predictions.json"),
    }


def render_response_str(pred: str):
    st.subheader("Response string")
    st.markdown(pred)
    st.markdown(f"Prediction: `{parse_prediction(pred)}`")


def render_steps_explicitly():
    pass


def assistant_api_error_debugger():
    data = load_math_dataset_ui()
    st.title("Assistant API error debugger")

    status = Counter([run["status"] for run in data["raw"].values()])
    st.sidebar.markdown(f"Assistant API status: `{status}`")

    # error status
    if st.checkbox("Debug error stats (this is expensive, )"):
        with st.form("f1"):
            if st.form_submit_button("Get failed case stats"):
                failed = []
                my_bar = st.progress(0, text="")

                for i, run in enumerate(data["raw"].values()):
                    if run["status"] == "failed":
                        failed.append(retrieve_run(run["thread_id"], run["run_id"]))
                    my_bar.progress((i + 1) / len(data["raw"]), text="")

                st.markdown(f"Failed case statistics: `{Counter([run.last_error.code for run in failed])}`")

        with st.form("f2"):
            if st.form_submit_button("Get in_progress case stats"):
                in_progress = []
                d = {}
                my_bar = st.progress(0, text="")

                for i, (k, run) in enumerate(data["raw"].items()):
                    if run["status"] == "in_progress":
                        res = retrieve_run(run["thread_id"], run["run_id"])
                        in_progress.append(res)
                        d[k] = res.status
                    my_bar.progress((i + 1) / len(data["raw"]), text="")

                st.markdown(f"in_progress case statistics: `{Counter([run.status for run in in_progress])}`")

                with st.expander("View each case"):
                    st.json(d)

    # in_progress = [run for run in data["raw"].values() if run["status"] == "in_progress"]

    # st.markdown(f"Failed case statistics: `{Counter([run['last_error']['code'] for run in failed])}`")

    st.header("Check each case")
    if status.keys() == {"success"}:
        st.title("All runs are successful.")
        return

    status = st.selectbox("Select status type", list(status.keys()))

    items = {k: v for k, v in data["raw"].items() if v["status"] == status}
    chosen_id = st.selectbox("Choose run ID", items.keys())

    st.subheader("Retrieve result")
    thread_id = items[chosen_id]["thread_id"]
    run_id = items[chosen_id]["run_id"]
    st.markdown(f"- thread_id: `{thread_id}`\n- run_id: `{run_id}`")

    run = retrieve_run(thread_id, run_id)
    st.json(run.model_dump())
    with st.expander("Render steps"):
        res = asyncio.run(
            parse_asssistant_steps(syncclient.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run_id, order="asc").data, complete_only=False)
        )
        st.markdown(res)

    render_response_str(data["str"][chosen_id])

    st.subheader("Cached assistant API response at the time of the error")
    st.json(items[chosen_id])


def app():
    st.sidebar.title("MATH run viewer")

    mode = st.sidebar.selectbox("Select mode", ["Assistant API error debugger"])

    if mode == "Assistant API error debugger":
        assistant_api_error_debugger()


if __name__ == "__main__":
    app()
