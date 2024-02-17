import asyncio
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads import Run
from openai.types.beta.threads.runs import CodeToolCall, MessageCreationStepDetails, RunStep, ToolCallsStepDetails

from src.openai_utils import LLMCache, client, syncclient

ASSISTANT_MAX_TIMEOUT = 300  # 5 mins
MAX_RETRY_COUNT = 5
WAIT_ON_ERROR = 5
WAIT_ON_RATELIMITERROR = 60  # 1 min, TPM resets every 60 seconds

KNOWN_ASSISTANT_IDS = {
    "gpt-4-codeinterpreter": "asst_vAQny8CJe4v3Bggbhmk1Hoc0",
    "gpt-3.5-codeinterpreter": "asst_GEgKcncwmDhp48s5Pt1uB4lc",
    "gpt4-csv": "asst_fRCYurcWYsdDen3JV62D7VOY",
    "gpt3.5-csv": "asst_nfhpN8nPMRCOGmC6SvPnCCQ8",
}


def _step_debug_message(step: RunStep) -> str:
    return f"RunStep(id={step.id}, run_id={step.run_id}, thread_id={step.thread_id}, step_details={step.step_details}, status={step.status})"


def _format_code_interpreter_result(tool_info: CodeToolCall) -> str:
    assert tool_info.type == "code_interpreter", f"Unexpected tool type, {tool_info}"
    assert len(tool_info.code_interpreter.outputs) == 1, f"Unexpected multiple outputs, {tool_info}"
    return f"```python\n{tool_info.code_interpreter.input}\n```\n\n{tool_info.code_interpreter.outputs[0].logs}"


async def parse_asssistant_steps(
    steps: list[RunStep], tool: bool = True, message_cache: Optional[dict] = None, record_message_mapping: bool = False, complete_only: bool = True
) -> str:
    """
    Parse assistant steps (client.beta.threads.runs.steps.list) and return a string.
    This implementation is based on the code interpreter tool.

    Args:
        steps (list[RunStep]): List of RunStep objects in ascending order. Acquired from `client.beta.threads.runs.steps.list(... order="asc")`.
        tool (bool): Whether to return tool trace.
        message_cache (Optional[dict]): Mapping from message ID to message content.
        record_message_mapping (bool): Whether to record message mapping. If True, discovered messages will be returned as a separate dict.
    Returns:
        str: Parsed string.
        Optional[dict]: If `record_message_mapping` is True, returns discovered message cache.
    """
    if message_cache is None:
        message_cache = {}
    if record_message_mapping:
        discovered_message_cache = {}

    trajectory = []

    for step in steps:
        if complete_only and step.status != "completed":
            continue

        if type(step.step_details) == MessageCreationStepDetails:
            # text message
            message_id = step.step_details.message_creation.message_id
            if message_id in message_cache:
                # use message_cache
                content = message_cache[message_id]
            else:
                message = await client.beta.threads.messages.retrieve(
                    thread_id=step.thread_id,
                    message_id=message_id,
                )
                content = message.content[0].text.value
                assert len(message.content) == 1, f"Incorrect message content, {_step_debug_message(step)}"
                if record_message_mapping:
                    discovered_message_cache[message_id] = content

            trajectory.append(content)

        elif type(step.step_details) == ToolCallsStepDetails:
            # tool
            if not tool:
                continue

            assert len(step.step_details.tool_calls) == 1, f"Unexpected multiple tool calls, {_step_debug_message(step)}"
            tool_info = step.step_details.tool_calls[0]

            if tool_info.type == "code_interpreter":
                trajectory.append(_format_code_interpreter_result(tool_info))  # RunStep
            else:
                raise NotImplementedError(f"Unknown tool call type, {_step_debug_message(step)}")

        else:
            raise ValueError(f"Unknown step type, {_step_debug_message(step)}")

    trajectory_str = "\n\n".join(trajectory)

    if record_message_mapping:
        return trajectory_str, discovered_message_cache
    else:
        return trajectory_str


@dataclass
class AssistantResponse:
    # AssistantResponse reporesnts the assistant output of a single `step` in a thread.
    thread_id: str
    run_id: str
    status: str

    # If not given, the following are computed in __post_init__ using the API. When loading from cache(AssistantCache),
    # these are already precomputed.
    steps: list[RunStep] = field(default=None)
    message_cache: dict = field(default=None)
    assistant_id: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.steps is None:
            self.steps = syncclient.beta.threads.runs.steps.list(thread_id=self.thread_id, run_id=self.run_id, order="asc").data

        if self.assistant_id is None:
            self.assistant_id = self.steps[0].assistant_id

        if self.message_cache is None:
            message_cache = {}
            for step in self.steps:
                if step.status != "completed":
                    continue
                if type(step.step_details) == MessageCreationStepDetails:
                    message = syncclient.beta.threads.messages.retrieve(
                        thread_id=step.thread_id,
                        message_id=step.step_details.message_creation.message_id,
                    )
                    assert len(message.content) == 1, f"Incorrect message content, {_step_debug_message(step)}"
                    message_cache[message.id] = message.content[0].text.value
            self.message_cache = message_cache

    async def to_string(self, tool: bool = True) -> str:
        """
        Get the most recent response.

        Args:
            tool (bool): Whether to return tool trace.
        """
        trajectory_str, discovered_message_cache = await parse_asssistant_steps(
            steps=self.steps, tool=tool, message_cache=self.message_cache, record_message_mapping=True
        )
        self.message_cache.update(discovered_message_cache)
        return trajectory_str

    def get_code(self) -> tuple[list[str], list[str]]:
        """
        Returns code intrepreter code and execution results.

        Returns:
            tuple: (list of code, list of results)
        """
        gen_code, execution_result = [], []

        for step in self.steps:
            if step.status != "completed":
                continue

            if type(step.step_details) == ToolCallsStepDetails:
                assert len(step.step_details.tool_calls) == 1, f"Unexpected multiple tool calls, {_step_debug_message(step)}"
                tool_info = step.step_details.tool_calls[0]
                if tool_info.type == "code_interpreter":
                    gen_code.append(tool_info.code_interpreter.input)
                    execution_result.append(tool_info.code_interpreter.outputs[0].logs)

        return gen_code, execution_result

    def serialize(self) -> dict:
        serialized = asdict(self)
        serialized["steps"] = [step.model_dump() for step in self.steps]
        return serialized

    @classmethod
    def from_dict(cls, serialized: dict) -> "AssistantResponse":
        """
        For loading can load serialized format
        response == AssistantResponse.from_dict(response.serialize())
        """
        serialized["steps"] = [RunStep(**step) for step in serialized["steps"]]
        return cls(**serialized)


class AssistantCache(LLMCache):
    def __init__(self, assistant_id: str, cache_dir: Optional[str] = "./.assistant_cache"):
        key_prefix = f"assistant<|SEP|>{assistant_id}<|SEP|>"
        super().__init__(cache_dir, key_prefix, meta={"assistant_id": assistant_id})

    def exists(self, prompt: object) -> bool:
        cache_dir = self.get_cache_dir(prompt)
        if not os.path.exists(cache_dir):
            return False

        with open(cache_dir, "rb") as f:
            cache = json.load(f)
        return cache["response"]["status"] == "completed"

    def _encode_response(self, response: AssistantResponse) -> str:
        return response.serialize()

    def _decode_response(self, response: dict) -> AssistantResponse:
        return AssistantResponse.from_dict(response)


async def _run_assistant_once(prompt: str, assistant_id: str, use_cache: bool = True, verbose: bool = False) -> AssistantResponse:
    thread: Thread = await client.beta.threads.create()

    await client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run: Run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    if verbose:
        logging.info(f"Run created. thread_id: {thread.id}, run_id: {run.id}")
    start_time = time.time()
    while time.time() < start_time + ASSISTANT_MAX_TIMEOUT:
        if verbose:
            logging.info(
                f"Polling run status. Status: {run.status}, thread_id: {thread.id}, run_id: {run.id}, time: {time.time() - start_time} seconds"
            )
        run: Run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        if run.status != "in_progress":  # failed or completed
            break
        await asyncio.sleep(5 + random.random())

    response = AssistantResponse(
        assistant_id=assistant_id,
        thread_id=thread.id,
        run_id=run.id,
        status=run.status,
    )
    return response, run


def is_rate_limit_error(run: Run) -> bool:
    return run.status == "failed" and run.error.code == "rate_limit_error"


async def run_assistant_once(
    prompt: str, assistant_name: Optional[str] = None, assistant: Optional[Assistant] = None, use_cache: bool = True, verbose: bool = False
) -> AssistantResponse:
    """
    Calls assistant API once and returns the response. A simpler interface like the completion API.

    Args:
        prompt (str): Prompt to send to the assistant.
        assistant_name (Optional[str]): Name of the assistant. If assistant is not provided, this must be provided.
        assistant (Optional[Assistant]): Assistant object. If provided, this will be used instead of assistant_name.
        use_cache (bool): Whether to load from local cache if the exact prompt is already computed previously. This should be False when running the same prompt multiple times with temperature. The results will be cached(saved) regardless of this condition.
        verbose (bool): Whether to print verbose logs.
    """
    if assistant is None:
        assert assistant_name is not None, f"assistant_name must be provided if assistant is not provided. list: {KNOWN_ASSISTANT_IDS.keys()}"
        assistant_id = KNOWN_ASSISTANT_IDS[assistant_name]
    else:
        assistant_id = assistant.id

    llmcache = AssistantCache(assistant_id=assistant_id)
    if use_cache and llmcache.exists(prompt):
        return llmcache.load(prompt)

    error_count = 0

    def should_retry() -> bool:
        nonlocal error_count

        if error_count > MAX_RETRY_COUNT:
            logging.warning(f"Failed to run assistant 3 times. Status: {run.status}, run_id: {run.id}")
            return False
        return True

    while True:
        response, run = await _run_assistant_once(prompt, assistant_id, use_cache, verbose)
        if run.status == "completed":
            break
        else:
            if is_rate_limit_error(run):
                asyncio.sleep(WAIT_ON_RATELIMITERROR)
            else:
                error_count += 1
                if error_count > MAX_RETRY_COUNT:
                    logging.warning(f"Run failed {error_count} times. Status: {run.status}, run_id: {run.id}")
                    break
                else:
                    if verbose:
                        logging.warning(f"Run failed {error_count} times, retrying. Status: {run.status}, run_id: {run.id}")
                    await asyncio.sleep(WAIT_ON_ERROR)

    # This can be confusing but it is intentional that llm cache is saved regardless of success.
    # First though they are recorded, failed runs will NOT be loaded from cache because the `AssistantCache.exists` method checks whether status is not "completed".
    # There are many reasons a run failed(e.g. timeout from running long loop, openai moderation API, ...).
    # The threads and runs are saved for debugging purposes of the error.
    llmcache.save(prompt, response)
    return response
