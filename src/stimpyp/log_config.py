import logging
from typing import Literal

__all__ = ['set_log_level']

LOG_LEVEL = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


def set_log_level(level: LOG_LEVEL):
    """Set logging level for stimpyp and its submodules.

    :param level: {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    """
    logger = logging.getLogger("stimpyp")
    numeric_level = getattr(logging, level, logging.INFO)
    logger.setLevel(numeric_level)
    for h in logger.handlers:
        h.setLevel(numeric_level)
