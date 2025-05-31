from __future__ import annotations

import abc
import re
import warnings
from pathlib import Path
from typing import Any, get_args, Literal, final

import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from typing_extensions import Self

from ._util import uglob
from .base import CAMERA_TYPE, AbstractLog
from .pyvstim import PyVlog
from .stimpy_core import RiglogData

__all__ = [
    'CAMERA_VERSION',
    'load_camlog',
    'AbstractCamlog',
    'LabCamlog',
    'PyCamlog'
]

CAMERA_VERSION = Literal['labcams', 'pycams']


def load_camlog(file: Path | str, *,
                camera_version: CAMERA_VERSION = 'labcams',
                suffix: str | None = None) -> LabCamlog | PyCamlog:
    """
    Read the camera recording log file

    :param file: log file for the camera
    :param camera_version: :class:`CAMERA_VERSION`
    :param suffix: suffix for the log file, If None then auto infer from :class:`CAMERA_VERSION`
    """
    file = Path(file)

    match camera_version:
        case 'labcams':
            return LabCamlog.load(file, suffix=suffix or '.camlog')
        case 'pycams':
            return PyCamlog.load(file, suffix=suffix or '.log')
        case _:
            raise ValueError(f'Unknown camera_version: {camera_version!r}. '
                             f'Expected one of {get_args(CAMERA_VERSION)}')


class AbstractCamlog(metaclass=abc.ABCMeta):
    """ABC for the camera logging information"""

    root: Path
    """tif root path"""
    frame_id: np.ndarray
    """frame index"""
    timestamp: np.ndarray
    """time stamp of each frame"""
    comment_info: dict[str, Any]
    """# information"""
    time_info: list[str]
    """i.e., # [21-03-02 15:23:08]. Note that time could be duplicated"""

    def __init__(self, root: Path,
                 frame_id: np.ndarray,
                 timestamp: np.ndarray,
                 comment_info: dict[str, Any],
                 time_info: list[str] | None = None):
        self.root = root
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.comment_info = comment_info
        self.time_info = time_info

    def __repr__(self):
        ret = pl.DataFrame().with_columns(
            pl.Series(self.frame_id).alias('frame_id'),
            pl.Series(self.timestamp).alias('timestamp')
        )
        return str(ret)

    @classmethod
    @abc.abstractmethod
    def load(cls, root: Path | str, suffix: str = '.camlog') -> Self:
        """
        load/parse the camera log file

        :param root: directory with camera log file
        :param suffix: file suffix
        :return: :class:`AbstractCamlog`
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _parse_comment(cls, content: str, comment_info, time_info) -> None:
        """update the comment_info attr"""
        pass

    @abc.abstractmethod
    def get_camera_time(self, riglog: RiglogData | PyVlog,
                        cam_name: str = '1P_cam') -> np.ndarray:
        pass

    # noinspection PyTypeChecker
    @property
    def nframes(self) -> int:
        """number of frames"""
        return self.frame_id[-1]

    def to_polars(self) -> pl.DataFrame:
        """convert :attr:`frame_id` and :attr:`timestamp` to polars DataFrame"""
        return pl.DataFrame({'frame_id': self.frame_id, 'time': self.timestamp})


# ======= #
# LabCams #
# ======= #

@final
class LabCamlog(AbstractCamlog):
    """Labcam log"""

    @classmethod
    def load(cls, root: Path | str,
             suffix: str = '.camlog') -> Self:

        root = Path(root)
        if root.is_dir():
            log_file = uglob(root, f'*{suffix}')
            root = root
        else:
            log_file = root
            root = root.parent

        comment_info = {}
        time_info = []
        log_data = []

        with log_file.open() as f:
            for line, content in enumerate(f):
                content = content.strip()
                cls._parse_comment(content, comment_info, time_info)
                if not content.startswith('#'):
                    log_data.append(list(map(float, content.split(','))))

            data = np.array(log_data)

            frame_id = data[:, 0].astype(int)
            timestamp = data[:, 1]

        return LabCamlog(root, frame_id, timestamp, comment_info, time_info)

    @classmethod
    def _parse_comment(cls, content: str, comment_info, time_info) -> None:

        header_pattern = r'#+ (?!.*?\[)#?([^:\n]+):\s*(.+)'
        m = re.match(header_pattern, content)

        if m:
            name, info = m.group(1), m.group(2)
            comment_info.update({f'{name}': info})

        event_pattern = r'# \[(.*?)\]\s*-\s*(.*)'
        if re.match(event_pattern, content):
            time_info.append(content)

    def get_camera_time(self, log: AbstractLog,
                        cam_name: CAMERA_TYPE = '1P_cam',
                        interpolate: bool = True) -> np.ndarray:
        """Interpolate camera log frames to those recorded by pyvstim.

        :param log: :class:`~stimpyp.base.AbstractLog`
        :param cam_name: camera name
        :param interpolate: Whether do the interpolation according to the number of log event
        :return: 1D camera time array in sec
        """
        if cam_name not in get_args(CAMERA_TYPE):
            raise ValueError(f'{cam_name} unknown')

        camera_event = log.camera_event[cam_name]
        cam_pulses = len(camera_event)

        if cam_pulses != self.frame_id[-1]:
            warnings.warn(f'Loss frame between riglog[{cam_pulses}] vs camlog[{self.frame_id[-1]}]')

        if interpolate:
            return interp1d(
                camera_event.value,
                camera_event.time,
                fill_value='extrapolate'
            )(self.frame_id)
        else:
            return camera_event.time


# ====== #
# PyCams #
# ====== #

@final
class PyCamlog(AbstractCamlog):
    """Pycams log"""

    @classmethod
    def load(cls, root: Path | str, suffix: str = '.log') -> Self:
        root = Path(root)
        if root.is_dir():
            log_file = uglob(root, f'*{suffix}')
            root = root
        elif root.is_file():
            log_file = root
            root = root.parent
        else:
            raise FileNotFoundError(f'{root} not found')

        #
        comment_info = {}
        log_data = []
        with log_file.open() as f:
            for line, content in enumerate(f):
                content = content.strip()
                cls._parse_comment(content, comment_info, [])
                if not content.startswith('#'):
                    log_data.append(list(map(float, content.split(','))))

            data = np.array(log_data)

            frame_id = data[:, 0].astype(int)
            timestamp = data[:, 1]

        return PyCamlog(root, frame_id, timestamp, comment_info, time_info=[])

    @classmethod
    def _parse_comment(cls, content: str, comment_info, time_info):
        match = re.match(r'#+ (.+?)\s*:\s*(.+)', content)
        if match:
            name, value = match.group(1), match.group(2)
            comment_info.update({name: value})

    def get_camera_time(self, riglog: RiglogData, cam_name: CAMERA_TYPE = '1P_cam'):
        raise NotImplementedError('pycams under dev')
