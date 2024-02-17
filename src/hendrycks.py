import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd

__all__ = ["is_equiv", "parse_prediction", "MathSample"]

RAW_DATA_PATH = "./data/data-raw/MATH"


class _Split(str, Enum):
    train = "train"
    test = "test"


class _Subject(str, Enum):
    algebra = "algebra"
    counting_and_probability = "counting_and_probability"
    geometry = "geometry"
    intermediate_algebra = "intermediate_algebra"
    number_theory = "number_theory"
    prealgebra = "prealgebra"
    precalculus = "precalculus"


@dataclass
class MathMeta:
    answer: str
    level: int
    subject: _Subject


@dataclass
class MathSample:
    # maybe implement some global interface for datset?
    id: str  # e.g. math/number_theory/764
    problem: str
    answer: str
    split: _Split
    meta: MathMeta

    def __post_init__(self):
        if isinstance(self.meta, dict):
            # lazy hack
            self.meta = MathMeta(**self.meta)

    def is_equiv(self, other: str, normalize=True):
        if normalize:
            other = parse_prediction(other)
        return is_equiv(self.meta.answer, other)

    @property
    def filepath(self):
        q_id = self.id.split("/")[-1]
        return Path(RAW_DATA_PATH) / self.split.value / self.meta.subject.value / f"{q_id}.json"


def load_hendrycks(
    split: Optional[Union[_Split, str]],
    subject: Optional[Union[_Subject, str]] = None,
    level: Optional[int] = None,
    base_filepath: Optional[str] = "data/math/math-all.jsonl",
) -> list[MathSample]:
    df = pd.read_json(base_filepath, lines=True)

    if isinstance(split, str) and split == "all":
        pass
    else:
        if isinstance(split, str):
            split = _Split(split)
        df = df[df["split"] == split.value]

    if subject is not None:
        if isinstance(subject, str):
            subject = _Subject(subject)
        df = df[df["meta"].apply(lambda x: x.get("subject") == subject.value)]

    if level is not None:
        df = df[df["meta"].apply(lambda x: x.get("level") == level)]

    ret = df.to_dict(orient="records")
    return [MathSample(**r) for r in ret]


def parse_prediction(s: str) -> str:
    if s is None:
        return None
    return _remove_boxed(_last_boxed_only_string(s))


def _remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # normalize text inside `boxed`

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    # compare answer strings inside `boxed`.
    # Raw model output should be processed using `parse_prediction` first. MathSample.is_equiv() does this automatically.
    if str1 is None and str2 is None:
        logging.warning("WARNING: Both None")
        return False
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
