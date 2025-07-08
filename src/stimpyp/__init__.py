import logging

from rich.logging import RichHandler

from .base import *
from .camlog import *
from .event import *
from .log_config import *
from .pyvstim import *
from .session import *
from .stimpy_core import *
from .stimpy_git import *
from .stimulus import *

logger = logging.getLogger("stimpyp")

if not logger.hasHandlers():
    handler = RichHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="[%H:%M:%S]"
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)  # default level
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
