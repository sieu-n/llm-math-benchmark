import json
from glob import glob
from pathlib import Path

import pandas as pd

from src.hendrycks import remove_boxed

FOLDER_PATH = "data/data-raw/MATH/test"
RESULT_PATH = "data/math-test.jsonl"

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
        data = {
            "id": make_id(path),
            "question": data["problem"],
            "answer": data["solution"],
            "meta": {
                "answer": remove_boxed(data["solution"]),
                "level": int(data["level"].split()[1]),
                "type": Path(path).parts[-2],
                "number": int(Path(path).stem),
            },
        }

    parsed_data.append(data)

pd.DataFrame(parsed_data).to_json(RESULT_PATH, orient="records", lines=True)
