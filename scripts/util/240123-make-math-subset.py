import numpy as np
import pandas as pd

np.random.seed(42)


base_dir = "data/math/math-test.jsonl"
RESULT_PATH = "data/math/math-test-subset.jsonl"
N = 3

df = pd.read_json(base_dir, lines=True)

df["subject"] = df["meta"].apply(lambda x: x["subject"])
df["level"] = df["meta"].apply(lambda x: x["level"])

samples = df.groupby(["subject", "level"]).apply(lambda x: x.sample(N)).reset_index(drop=True)

samples.drop(columns=["subject", "level"], inplace=True)

samples.to_json(RESULT_PATH, orient="records", lines=True)
