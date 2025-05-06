from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import Union, BinaryIO

__all__ = ['PathLike']

PathLike = Union[str, Path, bytes, BinaryIO, BufferedIOBase, BufferedReader]
