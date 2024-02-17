import json
from glob import glob
from pathlib import Path

import pandas as pd

from src.hendrycks import MathSample, parse_prediction

FOLDER_PATH = "data/data-raw/MATH/test"
RESULT_PATH = "data/math/math-test.jsonl"

paths = glob(FOLDER_PATH + "/*/*.json")


def make_id(path):
    p = Path(path)
    folder_name = p.parts[-2]
    file_name = p.stem
    return f"math/{folder_name}/{file_name}"


parsed_data = []
for path in paths:
    with open(path, "r") as f:
        data = json.load(f)
        try:
            level = data["level"].split()[1]

            if level == "?":
                level = -1
            else:
                level = int(level)
            data = MathSample(
                **{
                    "id": make_id(path),
                    "problem": data["problem"],
                    "answer": data["solution"],
                    "split": "test",  # "train" or "test
                    "meta": {
                        "answer": parse_prediction(data["solution"]),
                        "level": level,
                        "subject": Path(path).parts[-2],
                    },
                }
            )
        except Exception:
            breakpoint()

    parsed_data.append(data)

pd.DataFrame(parsed_data).to_json(RESULT_PATH, orient="records", lines=True)
