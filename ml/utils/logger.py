import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Colors:
    INFO = "\033[94m[INFO] "
    END = "\033[0m"
    ERROR = "\033[91m[ERROR] "


def get_color(msg_type):
    if msg_type == "INFO":
        return Colors.INFO
    elif msg_type == "ERROR":
        return Colors.ERROR
    elif msg_type == "END":
        return Colors.END


def print_msg(msg, msg_type):
    color = get_color(msg_type)
    msg = "".join([color, msg, Colors.END])
    print(msg)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")
