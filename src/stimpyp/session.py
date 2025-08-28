"""
User-specific experimental session
====================================

- Aim to know the time information about of a specific session in a given protocol


Pipeline
---------------

- Use protocol name to infer which type of protocol ``ProtocolAlias``
- Base on riglog/stimlog information to create a dictionary with ``Session``: ``SessionInfo`` pairs for a specific protocol
- Use the methods in ``SessionInfo`` to do the masking/slicing... with the ``RigEvent``

"""

from __future__ import annotations

import logging
from typing import NamedTuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AbstractStimlog
    from .stimpy_pygame import PyGameLinearStimlog

import numpy as np

__all__ = [
    'ProtocolAlias',
    'get_protocol_name',
    'get_protocol_sessions',
    #
    'Session',
    'SessionInfo'
]

ProtocolAlias = str
Session = str

logger = logging.getLogger(__name__)

# TODO user friendly
# TODO check numpy2.0 type

def get_protocol_name(log: 'Path | AbstractStimlog | PyGameLinearStimlog') -> ProtocolAlias:
    """dynamical protocol name inferred from filename.
    {visual_open_loop, light_dark_light, grey, vr}
    """
    from .base import AbstractStimlog
    from .stimpy_pygame import PyGameLinearStimlog

    if isinstance(log, (AbstractStimlog, PyGameLinearStimlog)):
        filename = log.riglog_data.riglog_file.stem
    else:
        filename = log.stem

    if 'ori' in filename:
        return 'visual_open_loop'
    elif '_LDL' in filename:
        return 'light_dark_light'
    elif '_black' in filename:
        return 'grey'
    elif '_linear' in filename:
        return 'vr'
    else:
        raise ValueError(f'unknown protocol filename:{filename}')


def get_protocol_sessions(stim: 'AbstractStimlog | PyGameLinearStimlog') -> list[SessionInfo]:
    alias = get_protocol_name(stim)
    if alias == 'visual_open_loop':
        return _get_protocol_vol(stim)
    elif alias == 'light_dark_light':
        return _get_protocol_ldl(stim)
    elif alias == 'grey':
        return _get_protocol_grey(stim)
    elif alias.startswith('vr'):
        return _get_protocol_vr(stim)
    else:
        raise RuntimeError(f'unknown alias: {alias}')


def _get_protocol_vol(stim: 'AbstractStimlog') -> list[SessionInfo]:
    """get session info for visual open loop protocol"""
    t0 = stim.riglog_data.exp_start_time
    t1 = stim.stim_start_time  # diode synced
    t2 = stim.stim_end_time  # diode synced
    t3 = stim.riglog_data.exp_end_time
    return [
        SessionInfo('light', (t0, t1)),
        SessionInfo('visual', (t1, t2)),
        SessionInfo('dark', (t2, t3)),
        SessionInfo('all', (t0, t3))
    ]


def _get_protocol_ldl(stim: 'AbstractStimlog') -> list[SessionInfo]:
    # diode signal is no longer reliable, use .prot file value instead
    from .stimpy_core import StimpyProtocol
    prot = StimpyProtocol.load(stim.stimlog_file.with_suffix('.prot'))

    t0 = stim.riglog_data.exp_start_time
    t1 = prot.start_blank_duration
    t2 = prot.total_duration - prot.end_blank_duration
    t3 = stim.riglog_data.exp_end_time

    return [
        SessionInfo('light_bas', (t0, t1)),
        SessionInfo('dark', (t1, t2)),
        SessionInfo('light_end', (t2, t3)),
        SessionInfo('all', (t0, t3)),
    ]


def _get_protocol_grey(stim: 'AbstractStimlog') -> list[SessionInfo]:
    t0 = stim.riglog_data.exp_start_time
    t1 = stim.riglog_data.exp_end_time
    return [SessionInfo('all', (t0, t1))]


def _get_protocol_vr(stim: 'PyGameLinearStimlog') -> list[SessionInfo]:
    try:
        t0 = stim.exp_start_time
        t1 = stim.passive_start_time
        t2 = stim.exp_end_time
    except AttributeError:
        raise RuntimeError('stimlog is not a PyGameLinearStimlog, forget to use --vr?')
    else:
        return [
            SessionInfo('close', (t0, t1)),
            SessionInfo('open', (t1, t2)),
            SessionInfo('all', (t0, t2))
        ]


# ============ #

class SessionInfo(NamedTuple):
    """session name and the corresponding time start-end"""

    name: Session
    """name of this session"""

    time: tuple[float, float]
    """time start/end of this session"""

    def time_mask_of(self, t: np.ndarray) -> np.ndarray:
        """
        create a mask for time array *t*.

        :param t: 1d time array
        :return: mask for this session
        """
        logger.info(f'time mask session: {self.name}, range: {self.time} sec')
        return np.logical_and(self.time[0] < t, t < self.time[1])

    def in_range(self, time: np.ndarray,
                 value: np.ndarray | None = None,
                 error: bool = True) -> tuple[Any, Any]:
        """
        Get the range (the first and last value) of value array in this session.

        :param time: 1d time array (T,)
        :param value: 1d value array. Shape should as same as *time* (T,)
        :param error: raise an error when empty.
        :return: tuple of first and last `value` or `time`.
        """
        x = self.time_mask_of(time)
        if value is not None:
            t = value[x]
        else:
            t = time[x]

        if len(t) == 0:
            if error:
                raise ValueError('empty in extracting value or time from time mask')
            return np.nan, np.nan

        return t[0], t[-1]

    def in_slice(self, time: np.ndarray,
                 value: np.ndarray,
                 error: bool = True) -> slice:
        """
        Get the slice of value in this session

        :param time: 1d time array (T,)
        :param value: 1d value array. Shape should as same as *time* (T,)
        :param error: raise an error when empty.
        :return: slice of `value`
        """
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError()

        v = value[self.time_mask_of(time)]

        if len(v) == 0:
            if error:
                raise ValueError('empty in extracting value or time from time mask')
            return slice(0, 0)

        return slice(int(v[0]), int(v[-1]))
