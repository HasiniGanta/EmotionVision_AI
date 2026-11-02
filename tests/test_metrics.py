import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import timer, format_ms, safe_divide


def test_timer_measures_real_elapsed_time():
    with timer() as t:
        time.sleep(0.01)
    assert t["elapsed_ms"] >= 9  # allow small scheduling variance


def test_format_ms_under_second():
    assert format_ms(500) == "500 ms"


def test_format_ms_over_second():
    assert format_ms(1500) == "1.50 s"


def test_safe_divide_normal():
    assert safe_divide(10, 2) == 5


def test_safe_divide_by_zero_returns_default():
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 0, default=-1) == -1
