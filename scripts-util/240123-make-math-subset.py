import numpy as np
import pandas as pd

np.random.seed(42)


base_dir = "data/math-test.jsonl"
RESULT_PATH = "data/math-test-subset.jsonl"
N = 3

df = pd.read_json(base_dir, lines=True)

df["type"] = df["meta"].apply(lambda x: x["type"])
df["level"] = df["meta"].apply(lambda x: x["level"])

samples = df.groupby(["type", "level"]).apply(lambda x: x.sample(N)).reset_index(drop=True)
samples[["id", "question", "answer", "meta"]].to_json(RESULT_PATH, orient="records", lines=True)
