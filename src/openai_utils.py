import abc
import datetime
import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import openai as _openai

openai = _openai

client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
syncclient = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def string_to_chat(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def is_chat_model(model: str) -> bool:
    # return True if model is a chat model, False otherwise
    chat_models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-1106",
        "gpt-4-turbo-preview",
        "gpt-4-0613",
        "gpt-4-1106-preview" "gpt-4-0125-preview",
        "gpt-4-vision-preview",
        "gpt-4",
    ]
    if model in chat_models:
        return True

    instructions_models = ["gpt-3.5-turbo-instruct"]
    if model in instructions_models:
        return False

    raise ValueError(f"Unknown model {model}")


class LLMCache(abc.ABC):
    def __init__(self, cache_dir: str, key_prefix: Optional[str] = "", meta: Optional[dict] = None) -> None:
        if meta is None:
            meta = {}

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.key_prefix = key_prefix
        self.meta = meta

    def get_cache_dir(self, prompt: object) -> str:
        prompt = self._prepare_hashkey(prompt)
        # python default `hash` is different per-session
        filename = Path(hashlib.sha256((self.key_prefix + str(prompt)).encode()).hexdigest() + ".json")
        return str(self.cache_dir / filename)

    def exists(self, prompt: object) -> bool:
        return os.path.exists(self.get_cache_dir(prompt))

    def save(self, prompt: object, response: object, meta: Optional[dict] = None) -> None:
        if meta is None:
            meta = {}

        assert "created_at" not in meta, "reserved key created_at already in meta"
        meta["created_at"] = str(datetime.datetime.now())

        filepath = self.get_cache_dir(prompt)
        assert "cache_dir" not in meta, "reserved key cache_dir already in meta"
        meta["cache_dir"] = filepath

        for k, v in self.meta.items():
            assert k not in meta, f"reserved key {k} already in meta"
            meta[k] = v

        data = {"prompt": prompt, "response": self._encode_response(response), "meta": meta}

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, prompt: object) -> object:
        filepath = self.get_cache_dir(prompt)
        with open(filepath, "rb") as f:
            data = json.load(f)
        return self._decode_response(data["response"])

    def _encode_response(self, response: object) -> object:
        # should return serializable object
        return response

    def _decode_response(self, response: object) -> object:
        return response

    def _prepare_hashkey(self, prompt: object) -> object:
        # serialize and normalize the prompt into a hashable object
        return json.dumps(prompt, sort_keys=True)


class OpenAICache(LLMCache):
    def __init__(self, model: str, cache_dir: Optional[str] = "./.openai_cache"):
        key_prefix = f"openai<|SEP|>{model}<|SEP|>"
        return super().__init__(cache_dir, key_prefix, meta={"model": model})


class OpenAICompletedStatus(Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    FAIL = "fail"


async def simple_openai_call(
    prompt: Union[str, list[dict]], model: Optional[str] = None, use_cache: bool = True, temperature: Optional[float] = 0, *args
) -> str:
    # string goes in, string comes out. For chat models, `prompt` is wrapped into [{"role": "user", "text": prompt}]
    llmcache = OpenAICache(model=model)
    if use_cache and llmcache.exists(prompt):
        return llmcache.load(prompt)

    if is_chat_model(model):
        # chat model (simple api)
        if isinstance(prompt, str):
            prompt = string_to_chat(prompt)
        response = await client.chat.completions.create(messages=prompt, model=model, temperature=temperature, *args).choices[0].message.content
    else:
        # completion model
        response = (
            await client.completions.create(
                prompt=prompt,
                model=model,
                temperature=temperature,
                *args,
            )
            .choices[0]
            .text
        )

    llmcache.save(prompt, response)
    return response
