from datetime import datetime


def ymdhms():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
