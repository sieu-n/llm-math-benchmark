import json
import os

import pytest

from src import openai_utils


def test_string_to_chat():
    assert openai_utils.string_to_chat("test") == [{"role": "user", "content": "test"}]


@pytest.mark.parametrize(
    "model, expected_output",
    [
        ("gpt-3.5-turbo", True),
        ("gpt-4", True),
        ("gpt-3.5-turbo-instruct", False),
    ],
)
def test_is_chat_model(model, expected_output):
    assert openai_utils.is_chat_model(model) == expected_output


def test_is_chat_model_error():
    with pytest.raises(ValueError):
        openai_utils.is_chat_model("gpt-5")


@pytest.mark.parametrize(
    "message, response, expected_cache_dir",
    [
        (
            [{"role": "user", "content": "I am human"}],
            "I am an AI",
            ".openai_cache/311ddfdda3fe578f4d2cdc0e1485fd93aa08992c44f69c3a810074c80654bafe.json",
        ),
        ("prompt", "response", ".openai_cache/dffe6dbe85b4b6d12b05ba0274d27656dd747637861b1f3a563990dcf6144031.json"),
    ],
)
def test_local_cache(message, response, expected_cache_dir):
    cache = openai_utils.OpenAICache(model="test-model")
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
    assert data["prompt"] == message
    assert data["response"] == response
    assert data["meta"]["model"] == "test-model"
    assert data["meta"]["cache_dir"] == cache_dir
    assert data["meta"]["meta"] == "data"
    assert "created_at" in data["meta"]


def test_cache_normalization():
    cache = openai_utils.OpenAICache(model="test-model")
    m1 = [{"role": "user", "content": "I am human"}, {"role": "assistant", "content": "I am AI"}]
    m2 = [{"content": "I am human", "role": "user"}, {"content": "I am AI", "role": "assistant"}]
    m3 = [{"role": "assistant", "content": "I am AI"}, {"role": "user", "content": "I am human"}]

    assert cache.get_cache_dir(m1) == cache.get_cache_dir(m2)
    assert cache.get_cache_dir(m1) != cache.get_cache_dir(m3)


@pytest.mark.expensive
@pytest.mark.parametrize(
    "message, model",
    [([{"role": "user", "content": "Hi"}], "gpt-3.5-turbo"), ("hi", "gpt-3.5-turbo"), ("1+1=", "gpt-3.5-turbo-instruct")],
)
def test_simple_openai_call(message, model):
    response = openai_utils.simple_openai_call(message, model=model, use_cache=False)
    assert isinstance(response, str)
