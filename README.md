# llm-math-benchmark




### dev

Setup

```
# (optional) venv
python -m venv env
source env/bin/activate

export PYTHONPATH=$(pwd)
pip install -r requirements.txt
pre-commit install
```

`ruff` and `pytest` is configured in `pre-commit`. 

```
(env) sieunpark@Sieunui-MacBookAir llm-math-benchmark % git commit -m "initial commit"
[WARNING] Unstaged files detected.
[INFO] Stashing unstaged files to /Users/sieunpark/.cache/pre-commit/patch1706343777-9616.
ruff.....................................................................Passed
ruff-format..............................................................Passed
pytest...................................................................Passed
```

TC with openai calls: there are some test cases that actually call openai. These are always marked `pytest.marker.expensive` and by default excluded when running `pytest` (through `addopts = -m "not expensive"` in the config toml). To invoke invoke these cases simply run `pytest m ''`


-----------

How to run in isolation.

```
ruff check --fix . 
pytest
```

Run pre-commit

```
git add .
pre-commit run
```