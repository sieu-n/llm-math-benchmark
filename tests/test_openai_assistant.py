import json
import os

import pytest
from openai.types.beta.threads.runs import ToolCallsStepDetails

from src import openai_assistant


@pytest.mark.parametrize(
    "message, response, expected_cache_dir",
    [
        (
            "use code to do blah blah",
            openai_assistant.AssistantResponse("dummy_thread_id", "dummy_run_id", "completed", [], {}, "dummy_assistant_id"),
            ".assistant_cache/b578e4c6f1074bf84e46c1894b87646bdbd33384c57e18978add28c053aee192.json",
        ),
    ],
)
def test_local_cache(message, response, expected_cache_dir):
    cache = openai_assistant.AssistantCache(assistant_id="dummy-assistant-id")
    cache_dir = cache.get_cache_dir(message)
    assert cache_dir == expected_cache_dir

    if os.path.exists(cache_dir):
        os.remove(cache_dir)

    cache.save(message, response, {"meta": "data"})
    assert os.path.exists(cache_dir)

    loaded_response = cache.load(message)
    assert loaded_response == response

    with open(cache_dir, "r") as f:
        data = json.load(f)
    assert data["meta"]["assistant_id"] == "dummy-assistant-id"
    assert data["meta"]["cache_dir"] == cache_dir
    assert data["meta"]["meta"] == "data"
    assert "created_at" in data["meta"]


CASE1 = dict(
    response_ids=dict(
        thread_id="thread_I6hJeO11QXtYvSLbwk0jfkdj",
        run_id="run_mkdB3ezE0Q1ljyN581evZGcw",
        status="completed",
        assistant_id="asst_vAQny8CJe4v3Bggbhmk1Hoc0",
    ),
    prompt="create an imaginatory pandas dataframe and print the head, then print some other statistics",
    response_notool="""The fictional pandas DataFrame was created with the following records:

```
           Name  Age         City  Salary
0    John Smith   28     New York   70000
1    Emma Jones   34  Los Angeles   82000
2  Alex Johnson   23      Chicago   54000
3     Mia Brown   45      Houston   92000
4  Olivia Davis   36      Phoenix   62000
```

Basic statistics derived from the numerical columns (`Age` and `Salary`) are:

```
             Age        Salary
count   5.000000      5.000000
mean   33.200000  72000.000000
std     8.348653  15231.546212
min    23.000000  54000.000000
25%    28.000000  62000.000000
50%    34.000000  70000.000000
75%    36.000000  82000.000000
max    45.000000  92000.000000
```

This summary includes the count, mean, standard deviation, minimum and maximum values, and the 25th, 50th (median), and 75th percentiles for both `Age` and `Salary`.""",
    response_tool="""```python
import pandas as pd

# Create a fictional DataFrame
data = {'Name': ['John Smith', 'Emma Jones', 'Alex Johnson', 'Mia Brown', 'Olivia Davis'],
        'Age': [28, 34, 23, 45, 36],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'Salary': [70000, 82000, 54000, 92000, 62000]}

df = pd.DataFrame(data)

# Print the head of the DataFrame
print("DataFrame Head:")
print(df.head())

# Print some basic statistics of the numerical columns
print("\\nBasic Statistics:")
print(df.describe())
```

DataFrame Head:
           Name  Age         City  Salary
0    John Smith   28     New York   70000
1    Emma Jones   34  Los Angeles   82000
2  Alex Johnson   23      Chicago   54000
3     Mia Brown   45      Houston   92000
4  Olivia Davis   36      Phoenix   62000

Basic Statistics:
             Age        Salary
count   5.000000      5.000000
mean   33.200000  72000.000000
std     8.348653  15231.546212
min    23.000000  54000.000000
25%    28.000000  62000.000000
50%    34.000000  70000.000000
75%    36.000000  82000.000000
max    45.000000  92000.000000


The fictional pandas DataFrame was created with the following records:

```
           Name  Age         City  Salary
0    John Smith   28     New York   70000
1    Emma Jones   34  Los Angeles   82000
2  Alex Johnson   23      Chicago   54000
3     Mia Brown   45      Houston   92000
4  Olivia Davis   36      Phoenix   62000
```

Basic statistics derived from the numerical columns (`Age` and `Salary`) are:

```
             Age        Salary
count   5.000000      5.000000
mean   33.200000  72000.000000
std     8.348653  15231.546212
min    23.000000  54000.000000
25%    28.000000  62000.000000
50%    34.000000  70000.000000
75%    36.000000  82000.000000
max    45.000000  92000.000000
```

This summary includes the count, mean, standard deviation, minimum and maximum values, and the 25th, 50th (median), and 75th percentiles for both `Age` and `Salary`.""",
)

# Example that runs code twice
CASE2 = dict(
    response_ids=dict(
        thread_id="thread_5YBoxlAzWMM1bjuYV6266CJg",
        run_id="run_kJ3gQMoOiJFc20QNRBS7CLsk",
        status="completed",
        assistant_id="asst_vAQny8CJe4v3Bggbhmk1Hoc0",
    ),
    prompt="""```
def reverse_list(lst):
    for i in range(len(lst)):
        temp = lst[i]
        lst[i] = lst[len(lst) - i]
        lst[len(lst) - i] = temp
    return lst

# Example usage
original_list = [1, 2, 3, 4, 5]
reversed_list = reverse_list(original_list)
print(reversed_list)
```

First, run this code. Then fix it and run it again. 
""",
    response_notool="""The code encountered an "IndexError" because it attempted to access an index that is out of range. This occurred due to a mistake in the index calculation. In Python, list indices start from 0, so when reversing the index should be offset by 1 more, for example, "lst[len(lst) - 1 - i]". Also, the loop should only iterate through half of the list to swap elements correctly. Let's fix the code and run it again.

After fixing the code, when reversing the original list `[1, 2, 3, 4, 5]`, the correctly reversed list is `[5, 4, 3, 2, 1]`. The corrected function now swaps elements properly and avoids "IndexError" by iterating only through the first half of the list and offsetting indices correctly.""",
    response_tool="""```python
def reverse_list(lst):
    for i in range(len(lst)):
        temp = lst[i]
        lst[i] = lst[len(lst) - i]
        lst[len(lst) - i] = temp
    return lst

# Example usage
original_list = [1, 2, 3, 4, 5]
reversed_list = reverse_list(original_list)
reversed_list
```

---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[1], line 10
      8 # Example usage
      9 original_list = [1, 2, 3, 4, 5]
---> 10 reversed_list = reverse_list(original_list)
     11 reversed_list

Cell In[1], line 4, in reverse_list(lst)
      2 for i in range(len(lst)):
      3     temp = lst[i]
----> 4     lst[i] = lst[len(lst) - i]
      5     lst[len(lst) - i] = temp
      6 return lst

IndexError: list index out of range


The code encountered an "IndexError" because it attempted to access an index that is out of range. This occurred due to a mistake in the index calculation. In Python, list indices start from 0, so when reversing the index should be offset by 1 more, for example, "lst[len(lst) - 1 - i]". Also, the loop should only iterate through half of the list to swap elements correctly. Let's fix the code and run it again.

```python
def reverse_list_fixed(lst):
    for i in range(len(lst) // 2):
        temp = lst[i]
        lst[i] = lst[-i - 1]
        lst[-i - 1] = temp
    return lst

# Example usage with a new list to avoid confusion
fixed_reversed_list = reverse_list_fixed([1, 2, 3, 4, 5])
fixed_reversed_list
```

[5, 4, 3, 2, 1]

After fixing the code, when reversing the original list `[1, 2, 3, 4, 5]`, the correctly reversed list is `[5, 4, 3, 2, 1]`. The corrected function now swaps elements properly and avoids "IndexError" by iterating only through the first half of the list and offsetting indices correctly.""",
)


@pytest.mark.expensive
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assistant_name, message, expected_output",
    [
        ("gpt-4-codeinterpreter", "What is the 100th fibonacci number? use code. The first 2 terms are 1, 1.", "354224848179261915075"),
        ("gpt-4-codeinterpreter", CASE1["prompt"], ""),
        ("gpt-4-codeinterpreter", CASE2["prompt"], "[5, 4, 3, 2, 1]"),
    ],
)
async def test_run_assistant_once(assistant_name, message, expected_output):
    response = await openai_assistant.run_assistant_once(message, assistant_name=assistant_name)

    # very non-deterministic because we can't set temperature, etc for assistant api.
    # https://community.openai.com/t/how-to-set-temperature-and-other-hyperparameters-of-model-in-open-ai-assistant-api/486368

    assert response.status == "completed"
    assert isinstance(response, openai_assistant.AssistantResponse)
    # assert response.ge

    traj = await response.to_string()
    assert expected_output in traj


@pytest.mark.sieun
@pytest.mark.asyncio
@pytest.mark.parametrize("case", [CASE1, CASE2])
async def test_parse_assistant_response(case: dict):
    response = openai_assistant.AssistantResponse(**case["response_ids"])

    # test the `string` methods
    message: str = await response.to_string(tool=False)
    assert case["response_notool"].strip() == message.strip()
    assert message == await openai_assistant.parse_asssistant_steps(response.steps, tool=False, message_cache=response.message_cache)

    message_with_tool: str = await response.to_string(tool=True)
    assert case["response_tool"].strip() == message_with_tool.strip()
    assert message_with_tool == await openai_assistant.parse_asssistant_steps(response.steps, tool=True, message_cache=response.message_cache)


@pytest.mark.parametrize(
    "kwargs, expected_output",
    [
        (
            {
                "tool_calls": [
                    {
                        "id": "abc",
                        "code_interpreter": {
                            "input": "# this is some Python code\nprint('hello world!')",
                            "outputs": [{"logs": "hello world!", "type": "logs"}],
                        },
                        "type": "code_interpreter",
                    }
                ],
                "type": "tool_calls",
            },
            "```python\n# this is some Python code\nprint('hello world!')\n```\n\nhello world!",
        ),
    ],
)
def test_format_code_interpreter_result(kwargs: dict, expected_output: str):
    tool_info = ToolCallsStepDetails(**kwargs).tool_calls[0]
    text = openai_assistant._format_code_interpreter_result(tool_info)

    assert text == expected_output
