import logging
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from neuralib.typing import PathLike
from stimpyp.event import RigEvent
from stimpyp.stimpy_core import RiglogData
from stimpyp.stimpy_git import load_stimlog

__all__ = ['pygame_position_event',
           'pygame_screen_event']

logger = logging.getLogger(__name__)


def pygame_position_event(file: PathLike,
                          rig: RiglogData | None = None, *,
                          offset_method: Literal['first', 'scalar', 'vector', 'none'] = 'none') -> RigEvent:
    if Path(file).suffix != '.stimlog':
        raise ValueError('File must end with .stimlog')

    stim = load_stimlog(file)
    pos = stim['Agent']['x'].to_numpy()
    pos_t = stim['Agent']['time'].to_numpy()

    match offset_method:
        case 'first':
            t = pos_t - pos_t[0]
            logger.info(f'offset first: {pos_t[0]} sec')
        case 'scalar':
            rig = _auto_rig_init(file) if rig is None else rig
            dt = _pygame_diode_offset_scalar(rig, stim)
            t = pos_t - dt
        case 'vector':
            rig = _auto_rig_init(file) if rig is None else rig
            t = _trial_time_factory(rig, stim)
        case 'none':
            t = pos_t
        case _:
            raise ValueError(f'Unknown offset method: {offset_method}')

    # offset boundary effect
    n_nan = np.isnan(t).sum()
    if n_nan > 0:
        logger.warning(f'nan observed during pygame task alignment: {n_nan}')

    valid = ~np.isnan(t)
    t, pos = t[valid], pos[valid]

    return RigEvent('position', np.vstack((t, pos)).T)


def _auto_rig_init(stimlog_file: PathLike, suffix: str = '.riglog') -> RiglogData:
    rig = Path(stimlog_file).with_suffix(suffix)
    if not rig.exists():
        raise FileNotFoundError(f'riglog file {rig} does not exist')
    return RiglogData(rig)


def _pygame_diode_offset_scalar(rig: RiglogData, stim: dict[str, pl.DataFrame]) -> float:
    riglog_time = rig.screen_event.time[0]
    stimlog_time = stim['Agent']['time'][0]
    dt = stimlog_time - riglog_time
    logger.info(f'offset as scalar: {dt} sec')
    return dt


def _pygame_diode_offset_vector(rig: RiglogData, stim: dict[str, pl.DataFrame]) -> np.ndarray:
    riglog_time = rig.screen_event.time[2::2]
    stimlog_time = stim['PhotoIndicator'].filter(pl.col('state') == True)['time'].to_numpy()[1:]
    dt = stimlog_time - riglog_time
    logger.info(f'offset as vector(trials): {dt} sec')
    return dt


def _trial_time_factory(rig: RiglogData, stim: dict[str, pl.DataFrame]) -> np.ndarray:
    agent_time = stim['Agent']['time'].to_numpy()
    trial_time = stim['LogDict']['time'][1:].to_numpy().copy()  # TODO fix?
    dt = _pygame_diode_offset_vector(rig, stim)

    # insert offset for segment before first PhotoIndicator
    dt = np.insert(dt, 0, dt[0])

    # replace first time with agent start time
    trial_time[0] = agent_time[0]

    # append end time for final segment
    trial_time = np.append(trial_time, agent_time[-1])

    corrected_time = np.full_like(agent_time, fill_value=np.nan, dtype=np.float64)
    for i, (t0, t1) in enumerate(zip(trial_time[:-1], trial_time[1:])):
        mx = np.logical_and(agent_time >= t0, agent_time < t1)
        corrected_time[mx] = agent_time[mx] - dt[i]

    return corrected_time


def pygame_screen_event(file: PathLike) -> RigEvent:
    if Path(file).suffix != '.stimlog':
        raise ValueError('File must end with .stimlog')

    stim = load_stimlog(file)
    t = stim['PhotoIndicator']['time']
    v = stim['PhotoIndicator']['state']

    return RigEvent('screen', np.vstack((t, v)).T)
