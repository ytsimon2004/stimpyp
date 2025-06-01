import logging

from rich.logging import RichHandler

from ._util import *
from .base import *
from .camlog import *
from .event import *
from .pyvstim import *
from .session import *
from .stimpy_core import *
from .stimpy_git import *
from .stimulus import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler()]
)
