from datetime import datetime


def set_seed():
    # huggingface set seed
    from transformers import set_seed

    set_seed(42)


def ymdhms():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
