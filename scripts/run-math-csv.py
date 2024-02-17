import asyncio
import json
import os
from functools import partial
from pathlib import Path
from typing import Annotated, List

import typer
from dotenv import load_dotenv
from InquirerPy import inquirer
from rich import print

from src import experiment, hendrycks
from src.aiojobs import run_batch_aiojobs
from src.openai_assistant import AssistantResponse, run_assistant_once

load_dotenv()

# CSV prompt
user_prompt = "The problem is: {problem}"


def get_save_path():
    p = Path("results/math-csv-repro") / f"exp-{experiment.ymdhms()}"
    os.makedirs(p, exist_ok=True)
    return p


async def run_inference_all_csvprompt(dataset: list[hendrycks.MathSample], debug=False):
    # prepare_prompt
    prompts: list[str] = [user_prompt.format(problem=sample.problem) for sample in dataset]
    call_openai_func = partial(run_assistant_once, assistant_name="gpt4-csv", use_cache=True)

    responses: list[AssistantResponse] = await run_batch_aiojobs(call_openai_func, args=prompts, limit=5)
    print("Inference done! Retrieving results...")

    # save results (retrieve strings calls openai api it's slow)
    async def retrieve_str_job(response: AssistantResponse) -> str:
        return await response.to_string()

    res_strings: list[str] = await run_batch_aiojobs(retrieve_str_job, args=responses, limit=100)
    predictions = [hendrycks.parse_prediction(res_string) for res_string in res_strings]

    save_path = get_save_path()
    print(f"Saving results in `{save_path}` ...")
    with open(save_path / "output-raw.json", "w") as f:
        json.dump({sample.id: response.serialize() for sample, response in zip(dataset, responses)}, f, indent=4)

    with open(save_path / "output-str.json", "w") as f:
        json.dump({sample.id: s for sample, s in zip(dataset, res_strings)}, f, indent=4)

    with open(save_path / "predictions.json", "w") as f:
        json.dump(
            {
                sample.id: {
                    "prediciton": pred,
                    "answer": sample.meta.answer,
                    "is_equiv": sample.is_equiv(pred, normalize=False),
                }
                for sample, pred in zip(dataset, predictions)
            },
            f,
            indent=4,
        )

    # accuracy
    correct = sum([sample.is_equiv(s) for sample, s in zip(dataset, res_strings)])
    success = sum([response.status == "completed" for response in responses])
    print(f"Accuracy (for success): {correct}/{success} ({correct/success*100:.2f}%)")
    print(f"Accuracy: {correct}/{len(dataset)} ({correct/len(dataset)*100:.2f}%)")

    if debug:
        for sample, response, res_string in zip(dataset, responses, res_strings):
            print("========================================")
            print(f"Problem: {sample.problem}")
            print("-------------------")
            print(f"Solution: {sample.answer}")
            print("-------------------")
            print(f"Response: {res_string}\n")
            print("-------------------")
            print(f"meta: {sample.meta}\n")
            print("prediction:", hendrycks.parse_prediction(res_string))
            print("is_equiv:", sample.is_equiv(res_string))
            breakpoint()

    return responses


def _main(
    subset: Annotated[
        str,
        typer.Option(help="Specify the subset: algebra, counting_and_probability, etc."),
    ] = None,
    split: Annotated[str, typer.Option(help='Specify the split to run: "train" or "test"')] = None,
    levels: Annotated[List[int], typer.Option(help="Specify one or more levels as a list.")] = None,
):
    """Solve MATH dataset problems using OpenAI assistant API.
    Arguments aren't required, the program will prompt for them if not provided.
    To specify multiple levels in your command, use --levels multiple times.
    (e.g. --levels 1 --levels 2 --levels 3)
    """
    subset = (
        subset
        or inquirer.select(
            message="Select the dataset subset to run with:",
            choices=[subset.value for subset in hendrycks._Subject],
        ).execute()
    )
    split = (
        split
        or inquirer.select(
            message="Select the dataset split to run with:",
            choices=[split.value for split in hendrycks._Split],
        ).execute()
    )
    levels = (
        levels
        or inquirer.select(
            message="Select one or more difficulty levels:",
            choices=[1, 2, 3, 4, 5],
            multiselect=True,
        ).execute()
    )

    dataset = []

    for level in levels:
        dataset.append(hendrycks.load_hendrycks(subject=subset, split=split, level=level))
    asyncio.run(run_inference_all_csvprompt(sum(dataset, [])))


if __name__ == "__main__":
    # typer.run(_main)

    dataset: list[hendrycks.MathSample] = hendrycks.load_hendrycks(split="test", subject="intermediate_algebra", level=4) + hendrycks.load_hendrycks(
        split="test", subject="intermediate_algebra", level=5
    )
    print(len(dataset))
    res = asyncio.run(run_inference_all_csvprompt(dataset))
