import logging
import time
from contextlib import contextmanager

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _configured = True
    return logging.getLogger(name)


@contextmanager
def timed(timings: dict, key: str):
    """Record wall-clock milliseconds of a block into ``timings[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round((time.perf_counter() - start) * 1000, 2)
