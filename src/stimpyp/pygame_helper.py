import logging
from typing import Literal

import numpy as np
import polars as pl

from stimpyp._type import PathLike
from stimpyp.event import RigEvent
from stimpyp.stimpy_core import RiglogData
from stimpyp.stimpy_git import load_stimlog

__all__ = ['PyGameLinearStimlog']

logger = logging.getLogger(__name__)

DIODE_OFFSET_TYPE = Literal['scalar', 'vector']


class PyGameLinearStimlog:

    def __init__(self, riglog: RiglogData | None,
                 filepath: PathLike | None = None,
                 diode_offset: bool = True,
                 offset_method: DIODE_OFFSET_TYPE = 'scalar'):
        if riglog is None and filepath is None:
            raise TypeError("Either riglog or filepath must be specified")

        self.riglog_data = riglog
        self.stimlog_data = load_stimlog(riglog.stimlog_file)

        # diode offset
        self.diode_offset = diode_offset
        self.offset_method = offset_method

    @property
    def virtual_position_event(self) -> RigEvent:
        df = self.get_agent_dataframe()
        return RigEvent('position', np.vstack((df['time'], (df['x']))).T)

    @property
    def screen_event(self) -> RigEvent:
        df = self.get_photo_indicator_dataframe()
        return RigEvent('screen', np.vstack((df['time'], (df['state']))).T)

    def get_agent_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['Agent'])

    def get_photo_indicator_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['PhotoIndicator'])

    def get_log_dict_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['LogDict'])

    def get_state_machine_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['DispatchStateMachine'])

    def _offset(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.diode_offset:
            match self.offset_method:
                case 'scalar':
                    dt = _diode_offset_scalar(self.riglog_data, self.stimlog_data)
                    return df.with_columns((pl.col('time') - dt).alias('time'))
                case 'vector':
                    return self._trial_base_offset(df)
                case _:
                    raise ValueError('Unknown offset method')
        return df

    def _trial_base_offset(self, df: pl.DataFrame) -> pl.DataFrame:
        event_time = df['time'].to_numpy()
        trial_time = self.stimlog_data['LogDict']['time'][1:].to_numpy().copy()
        dt = _diode_offset_vector(self.riglog_data, self.stimlog_data)

        # insert offset for segment before first PhotoIndicator
        dt = np.insert(dt, 0, dt[0])

        # replace first time with agent start time
        trial_time[0] = event_time[0]

        # append end time for final segment
        trial_time = np.append(trial_time, event_time[-1])

        corrected_time = np.full_like(event_time, fill_value=np.nan, dtype=np.float64)
        for i, (t0, t1) in enumerate(zip(trial_time[:-1], trial_time[1:])):
            mx = np.logical_and(event_time >= t0, event_time < t1)
            corrected_time[mx] = event_time[mx] - dt[i]

        # boundary effect
        n_nan = np.isnan(corrected_time).sum()
        if n_nan > 0:
            logger.warning(f'nan observed during pygame task alignment: {n_nan}')

        return df.with_columns(pl.Series('time', corrected_time))

    def get_max_virtual_length(self) -> np.ndarray:
        """`Array[float, L]`"""
        return self.get_agent_dataframe().filter(pl.col('touch') == 'reward')['x'].to_numpy()

    def get_virtual_length(self, count: int = 1, length: float = 150) -> float:
        """get actual virtual trial length based on riglog encoder value mapping

        :param count: number of photosensing for each trial for rig encoder
        :param length: length in cm for each trial for rig encoder
        """
        if not self.diode_offset:
            raise ValueError('diode offset need to be set')

        reward = self.get_agent_dataframe().filter(pl.col('touch') == 'reward')

        rig = self.riglog_data
        rig_pos = rig.position_event
        unwarp_enc = rig.unwarp_circular_position()

        # use first lap time
        mx = rig_pos.time < reward['time'][0]
        ret = unwarp_enc[mx]

        f, _ = self.riglog_data.get_encoder_factor(count, length)

        return (ret.max() - ret.min()) * f


def _diode_offset_scalar(rig: RiglogData, stim: dict[str, pl.DataFrame]) -> float:
    riglog_time = rig.screen_event.time[1]
    stimlog_time = stim['PhotoIndicator']['time'][2]
    dt = stimlog_time - riglog_time
    logger.info(f'offset as scalar: {dt} sec')
    return dt


def _diode_offset_vector(rig: RiglogData, stim: dict[str, pl.DataFrame]) -> np.ndarray:
    riglog_time = rig.screen_event.time[2::2]
    stimlog_time = stim['PhotoIndicator'].filter(pl.col('state') == True)['time'].to_numpy()[1:]
    dt = stimlog_time - riglog_time
    logger.info(f'offset as vector(trials): {dt} sec')
    return dt
