import pickle
from datetime import datetime

import pandas as pd
from openai_utils import OpenAI
from tqdm import tqdm

EXP_NAME = "240123-baseline-cot-gpt4"

client = OpenAI()


def make_prompt(sample: dict):
    question = sample["question"]
    return [
        {
            "role": "user",
            "content": f"Solve the problem and put your answer in \\boxed{{}}. The problem is: {question}.",
        }
    ]


def parse_result(response: dict):
    return response["choices"][0]["text"]


# openai call
data = pd.read_json("data/math-test-subset.jsonl", lines=True)

prompts = [make_prompt(sample) for sample in data.to_dict(orient="records")]

response = []

for sample in tqdm(prompts):
    response.append(client.chat.completions.create(model="gpt-4-1106-preview", messages=sample).choices[0].message.content)

# save pkl file in case exception.
with open("temp/response.pkl", "wb") as f:
    pickle.dump(response, f)

result = []
for i, sample in data.iterrows():
    result.append(
        {
            "id": sample["id"],
            "response": response[i],
        }
    )

with open(f"data/runs/{EXP_NAME}.pkl", "wb") as f:
    pickle.dump({"name": EXP_NAME, "time": datetime.now(), "out": result}, f)
