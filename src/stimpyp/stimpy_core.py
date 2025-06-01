from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, final, Iterable

import numpy as np
import polars as pl
from typing_extensions import Self

from ._type import PathLike
from ._util import unfold_stimuli_condition, try_casting_number, cls_hasattr
from .base import AbstractLog, AbstractStimlog, AbstractStimProtocol, AbstractStimulusPattern
from .session import Session, SessionInfo, get_protocol_sessions
from .stimulus import GratingPattern, FunctionPattern

__all__ = [
    'load_riglog',
    'RiglogData',
    #
    'Stimlog',
    #
    'load_protocol',
    'StimpyProtocol'
]

logger = logging.getLogger(__name__)


def load_riglog(root_path: PathLike,
                diode_offset: bool = True,
                reset_mapping: dict[int, list[str]] | None = None) -> RiglogData:
    """
    load riglog data

    :param root_path: riglog file path or riglog directory path
    :param diode_offset: do the diode offset to sync the time between riglog and stimlog
    :param reset_mapping:
    :return: :class:`RiglogData`
    """
    return RiglogData(root_path=root_path, diode_offset=diode_offset, reset_mapping=reset_mapping)


@final
class RiglogData(AbstractLog):
    """class for handle the riglog file for stimpy **bitbucket/github** version
    (mainly tested in the commits derived from master branch)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, log_suffix='.riglog', **kwargs)

        self.__stimlog_cache: AbstractStimlog | None = None
        self.__prot_cache: StimpyProtocol | None = None

    @classmethod
    def _cache_asarray(cls, filepath: Path, square_brackets: bool) -> np.ndarray:
        output = filepath.with_name(filepath.stem + '_riglog.npy')

        if not output.exists():
            if square_brackets:
                riglog = np.loadtxt(
                    filepath,
                    delimiter=',',
                    comments=['#', 'None'],
                    converters={
                        0: lambda it: float(it[1:]),
                        3: lambda it: float(it[:-1])
                    },
                )
                logger.debug('square bracket were found in riglog file')
            else:
                riglog = pl.read_csv(filepath, comment_prefix='#').to_numpy()

            np.save(output, riglog)

        return np.load(output)

    def with_sessions(self, session: Session | tuple[Session, ...]) -> Self:
        """
        Truncate the instance `dat` with the given session(s)

        :param session: session name(s)
        :return:
        """
        dy = self.get_stimlog().session_trials()

        if isinstance(session, str):
            t0, t1 = dy[session].time
        elif isinstance(session, tuple):
            t_all = np.array([dy[s].time for s in session])
            t0, t1 = np.min(t_all), np.max(t_all)
            logger.info(f'get trange from multiple sessions, {session}: t0:{t0}, t1:{t1}')
        else:
            raise TypeError('')

        t = self.dat[:, 2] / 1000
        mask = np.logical_and(t0 < t, t < t1)
        self.dat = self.dat[mask]

        return self

    @property
    def stimlog_file(self) -> Path:
        return self.riglog_file.with_suffix('.stimlog')

    def get_stimlog(self, csv_output: bool = True) -> AbstractStimlog:
        """
        Initialize the stimlog instance

        :param csv_output: if stimlog is exported to separated csv file
        """
        match self.__stimlog_cache, self.version:
            case (None, 'stimpy-git'):
                from .stimpy_git import StimlogGit
                self.__stimlog_cache = StimlogGit(
                    self,
                    self.stimlog_file,
                    self._reset_mapping,
                    csv_output=csv_output,
                    diode_offset=self._diode_offset
                )
            case (None, 'stimpy-bit'):
                self.__stimlog_cache = Stimlog(self, self.stimlog_file, self._reset_mapping, self._diode_offset)
            case (None, _):
                raise ValueError(f'unknown version: {self.version}')

        logger.debug(f'init stimlog with {type(self.__stimlog_cache).__name__}')
        return self.__stimlog_cache

    def get_protocol(self) -> 'StimpyProtocol':
        if self.__prot_cache is None:
            self.__prot_cache = StimpyProtocol.load(self.prot_file)

        return self.__prot_cache


@final
class Stimlog(AbstractStimlog):
    """class for handle the stimlog file for stimpy **bitbucket** version
    (mainly tested in the commits derived from master branch)

    `Dimension parameters`:

        N = numbers of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        P = number of acquisition sample pulse (Visual parameters)

        M = number of statemachine pulse
    """

    config: dict[str, Any] = {}
    """i.e., name, commit hash, missed_frames, ..."""

    log_info: dict[int, str] = {}
    """i.e., {10: 'vstim', 20: 'stateMachine'}"""

    log_header: dict[int, list[str]] = {}
    """i.e., {10: ['code','presentTime','iStim', ...], 20: ['code', 'elapsed', 'cycle', ...]}"""

    photo_state: np.ndarray
    """photo diode on-off. Array[int, P]. value domain in (0,1)"""

    indicator_flag: np.ndarray
    """accumulated and zero foreach stim. Array[int, P]. value domain in (1, ...)"""

    # ========================================= #
    # StateMachine logging (start from code 20) #
    # ========================================= #

    s_on_v: np.ndarray
    """statemachine sync time on v, use for time sync if diode signal not reliable. Array[float, M]"""

    state_time: np.ndarray
    """Array[float, M]"""

    state_cycle: np.ndarray
    """Array[float, M]"""

    state_new_state: np.ndarray
    """Array[float, M]"""

    state_old_state: np.ndarray
    """Array[float, M]"""

    state_elapsed: np.ndarray
    """Array[float, M]"""

    state_trial_type: np.ndarray
    """Array[float, M]"""

    def __init__(self,
                 riglog: RiglogData,
                 file_path: PathLike,
                 reset_mapping: dict[int, list[str]] | None = None,
                 diode_offset: bool = True,
                 sequential_offset: bool = True):
        """
        :param riglog: ``RiglogData``
        :param file_path: stimlog filepath
        :param diode_offset: If do the diode offset to sync the time to riglog
        :param sequential_offset: do the sequential offset to sync time with :class:`RiglogData`
        """

        super().__init__(riglog, file_path, reset_mapping)
        self.diode_offset = diode_offset

        # diode
        self._do_sequential_diode_offset = sequential_offset
        self._cache_time_offset: float | np.ndarray | None = None

        self._reset()

    def _reset(self):
        try:
            stim_type = self.riglog_data.get_stimulus_type()
        except AttributeError:
            print(f'no riglog init, for only testing, some methods might causes problem', vtype='warning')
            return self._reset_gratings()  # testing

        match stim_type, self._reset_mapping:
            case ('gratings', None):
                self._reset_gratings()
                return None
            case ('functions', None):
                self._reset_functions()
                return None
            case (_, mapping) if mapping is not None:
                self._reset_cust_mapping()
                return None
            case _:
                raise NotImplementedError('')

    def _eager_comment_info(self):
        """Eager get the #comment_info"""
        with self.stimlog_file.open() as _f:
            for number, line in enumerate(_f):
                line = line.strip()

                if len(line) == 0:
                    continue

                if line.startswith('#'):
                    self._reset_comment_info(line)

    def _reset_cust_mapping(self):
        def _cast(s) -> int | float:
            try:
                v = float(s)
                if v.is_integer():
                    return int(v)
                else:
                    return v
            except (ValueError, IndexError):
                return -1

        self._eager_comment_info()
        vlog = [[] for _ in range(len(self.log_header[10]) - 1)]  # without code
        state = [[] for _ in range(len(self.log_header[20]) - 1)]  # without code

        with self.stimlog_file.open() as _f:
            for number, line in enumerate(_f):
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                part = line.split(',')
                try:
                    code = int(part[0])
                    if code == 10:
                        for i in range(len(vlog)):
                            try:
                                vlog[i].append(_cast(part[i + 1]))
                            except IndexError:
                                vlog[i].append(-1)
                    elif code == 20:
                        for i in range(len(state)):
                            state[i].append(_cast(part[i + 1]))
                    else:
                        raise ValueError(f'unknown code: {code}')

                except BaseException as e:
                    raise RuntimeError(f'error parsing line: {line} in line:{number}, {repr(e)}')

        #
        for code, fields in self._reset_mapping.items():
            if code == 10:
                for i, f in enumerate(fields):
                    setattr(self, f, np.array(vlog[i]))
            elif code == 20:
                for i, f in enumerate(fields):
                    v = np.array(state[i])
                    if f in ('state_time', 'state_elapsed'):  # to sec
                        v = v / 1000
                    setattr(self, f, v)
            else:
                raise ValueError(f'unknown code: {code}')

        self._check_mapping()

    def _check_mapping(self) -> None:
        for code, fields in self._reset_mapping.items():
            # length check
            if len(fields) != len(self.log_header[code]) - 1:
                raise ValueError(f'reset mapping number mismatch: should follow the order: {self.log_header[code]}')

            # name check
            for f in fields:
                if not cls_hasattr(Stimlog, f):
                    raise ValueError(f'reset mapping need to be the one of the annotations: "{f}"')

    def _reset_gratings(self):
        time = []
        stim_index = []
        trial_index = []
        photo_state = []
        contrast = []
        ori = []
        sf = []
        phase = []
        frame_index = []

        s_on_v = []
        s_present_time = []
        s_cycle = []
        s_new_state = []
        s_old_state = []
        s_state_elapsed = []
        s_trial_type = []

        with self.stimlog_file.open() as _f:
            for number, line in enumerate(_f):
                line = line.strip()
                if len(line) == 0:
                    continue

                if line.startswith('#'):
                    self._reset_comment_info(line)
                    continue

                part = line.split(',')

                try:
                    code = int(part[0])

                    if code == 10:
                        time.append(float(part[1]))

                        try:
                            stim_value = int(part[2])
                        except ValueError:  # 'None'
                            stim_value = -1
                        stim_index.append(stim_value)

                        trial_index.append(int(part[3]))
                        photo_state.append(int(part[4]))

                        if len(part) > 5:
                            contrast.append(float(part[5]))
                            ori.append(int(float(part[6])))
                            sf.append(float(part[7]))
                            phase.append(float(part[8]))
                            frame_index.append(int(part[9]))
                        else:
                            # empty list item = -1
                            contrast.append(-1)
                            ori.append(-1)
                            sf.append(-1)
                            phase.append(-1)
                            frame_index.append(-1)

                    elif code == 20:
                        s_on_v.append(time[-1])  # sync s to v
                        s_present_time.append(int(part[1]) / 1000)
                        s_cycle.append(int(part[2]))
                        s_new_state.append(int(part[3]))
                        s_old_state.append(int(part[4]))
                        s_state_elapsed.append(int(part[5]) / 1000)
                        s_trial_type.append(int(part[6]))

                    else:
                        raise ValueError(f'unknown code : {code}')

                except BaseException as e:
                    raise RuntimeError(f'line {number}: {line}') from e

        self.time = np.array(time)
        self.stim_index = np.array(stim_index)
        self.trial_index = np.array(trial_index)
        self.photo_state = np.array(photo_state)
        self.contrast = np.array(contrast)
        self.ori = np.array(ori)
        self.sf = np.array(sf)
        self.phase = np.array(phase)
        self.frame_index = np.array(frame_index)

        # special case
        self._reset_tf()

        # state machine
        self.s_on_v = np.array(s_on_v)
        self.state_time = np.array(s_present_time)
        self.state_cycle = np.array(s_cycle)
        self.state_new_state = np.array(s_new_state)
        self.state_old_state = np.array(s_old_state)
        self.state_elapsed = np.array(s_state_elapsed)
        self.state_trial_type = np.array(s_trial_type)

    def _reset_functions(self):
        time = []
        stim_index = []
        trial_index = []
        photo_state = []
        contrast = []
        pos_x = []
        pos_y = []
        size_x = []
        size_y = []
        frame_index = []

        s_on_v = []
        s_present_time = []
        s_cycle = []
        s_new_state = []
        s_old_state = []
        s_state_elapsed = []
        s_trial_type = []

        with self.stimlog_file.open() as _f:
            for number, line in enumerate(_f):
                line = line.strip()
                if len(line) == 0:
                    continue

                if line.startswith('#'):
                    self._reset_comment_info(line)
                    continue

                part = line.split(',')

                try:
                    code = int(part[0])

                    if code == 10:
                        time.append(float(part[1]))

                        try:
                            stim_value = int(part[2])
                        except ValueError:  # 'None'
                            stim_value = -1
                        stim_index.append(stim_value)

                        trial_index.append(int(part[3]))
                        photo_state.append(int(part[4]))

                        if len(part) > 5:
                            contrast.append(float(part[5]))
                            pos_x.append(int(float(part[6])))
                            pos_y.append(float(part[7]))
                            size_x.append(float(part[8]))
                            size_y.append(int(part[9]))
                            frame_index.append(int(part[10]))
                        else:
                            # empty list item = -1
                            contrast.append(-1)
                            pos_x.append(-1)
                            pos_y.append(-1)
                            size_x.append(-1)
                            size_y.append(-1)
                            frame_index.append(-1)

                    elif code == 20:
                        s_on_v.append(time[-1])  # sync s to v
                        s_present_time.append(int(part[1]) / 1000)
                        s_cycle.append(int(part[2]))
                        s_new_state.append(int(part[3]))
                        s_old_state.append(int(part[4]))
                        s_state_elapsed.append(int(part[5]) / 1000)
                        s_trial_type.append(int(part[6]))

                    else:
                        raise ValueError(f'unknown code : {code}')

                except BaseException as e:
                    raise RuntimeError(f'line {number}: {line}') from e

        self.time = np.array(time)
        self.stim_index = np.array(stim_index)
        self.trial_index = np.array(trial_index)
        self.photo_state = np.array(photo_state)
        self.contrast = np.array(contrast)
        self.pos_x = np.array(pos_x)
        self.pos_y = np.array(pos_y)
        self.size_x = np.array(size_x)
        self.size_y = np.array(size_y)
        self.frame_index = np.array(frame_index)

        # state machine
        self.s_on_v = np.array(s_on_v)
        self.state_time = np.array(s_present_time)
        self.state_cycle = np.array(s_cycle)
        self.state_new_state = np.array(s_new_state)
        self.state_old_state = np.array(s_old_state)
        self.state_elapsed = np.array(s_state_elapsed)
        self.state_trial_type = np.array(s_trial_type)

    def _reset_tf(self):
        try:
            prot = self.riglog_data.get_protocol()
        except AttributeError:
            warnings.warn('cannot inferred prot file due to class not has riglog attribute', vtype='warning')
            pass
        else:
            v_start = self.frame_index == 1
            nr = self.stim_index[v_start]
            self.tf = prot.tf[nr]

    def _reset_comment_info(self, line: str):
        if 'CODES' in line:
            heading, content = line.split(': ')
            iter_codes = content.split(',')
            for pair in iter_codes:
                code, num = pair.split('=')
                code = code.strip()
                value = int(num.strip())
                self.log_info[value] = code
            logger.debug(f'Parsed log_info: {self.log_info}')

        elif 'VLOG HEADER' in line:
            heading, content = line.split(':')
            self.log_header[10] = content.split(', ')
            logger.debug(f'Parsed VLOG HEADER: {self.log_header[10]}')

        elif 'STATE HEADER' in line:
            heading, content = line.split(': ')
            self.log_header[20] = content.split(',')
            logger.debug(f'Parsed STATE HEADER: {self.log_header[20]}')

        elif 'Commit' in line:
            heading, content = line.split(': ')
            commit = content.strip()
            self.config['commit_hash'] = commit
            logger.debug(f'Parsed commit hash: {commit}')

        elif 'Missed' in line:
            match = re.search(r'\d+', line)
            if match:
                missed = int(match.group())
            else:
                missed = None
            self.config['missed_frames'] = missed
            logger.debug(f'Parsed missing frame: {missed}')

    def get_visual_stim_dataframe(self, stim_only: bool = True) -> pl.DataFrame:
        """
        Get the stimlog visual stimulation logging as dataframe::

            ┌─────────────┬───────┬────────┬───────┬───┬───────┬──────┬───────────┬──────────┐
            │ presentTime ┆ iStim ┆ iTrial ┆ photo ┆ … ┆ ori   ┆ sf   ┆ phase     ┆ stim_idx │
            │ ---         ┆ ---   ┆ ---    ┆ ---   ┆   ┆ ---   ┆ ---  ┆ ---       ┆ ---      │
            │ f64         ┆ f64   ┆ f64    ┆ f64   ┆   ┆ f64   ┆ f64  ┆ f64       ┆ f64      │
            ╞═════════════╪═══════╪════════╪═══════╪═══╪═══════╪══════╪═══════════╪══════════╡
            │ 904.048038  ┆ 69.0  ┆ 0.0    ┆ 0.0   ┆ … ┆ 270.0 ┆ 0.16 ┆ 0.066667  ┆ 1.0      │
            │ …           ┆ …     ┆ …      ┆ …     ┆ … ┆ …     ┆ …    ┆ …         ┆ …        │
            │ 2710.807378 ┆ 45.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 270.0 ┆ 0.04 ┆ 11.933333 ┆ 179.0    │
            │ 2710.824102 ┆ 45.0  ┆ 4.0    ┆ 0.0   ┆ … ┆ 270.0 ┆ 0.04 ┆ 11.933333 ┆ 179.0    │
            └─────────────┴───────┴────────┴───────┴───┴───────┴──────┴───────────┴──────────┘

        :param stim_only: only show the stimulation epoch
        :return: visual stimuli dataframe
        """
        stim_type = self.riglog_data.get_stimulus_type()
        headers = self.log_header[10][1:]
        mask = self.frame_index != -1 if stim_only else slice(None, None)

        match stim_type, self._reset_mapping:
            case ('gratings', None):
                return pl.DataFrame(
                    np.vstack([self.time[mask],
                               self.stim_index[mask],
                               self.trial_index[mask],
                               self.photo_state[mask],
                               self.contrast[mask],
                               self.ori[mask],
                               self.sf[mask],
                               self.phase[mask],
                               self.frame_index[mask]]),
                    schema=headers
                )

            case ('functions', None):
                return pl.DataFrame(
                    np.vstack([self.time[mask],
                               self.stim_index[mask],
                               self.trial_index[mask],
                               self.photo_state[mask],
                               self.contrast[mask],
                               self.pos_x[mask],
                               self.pos_y[mask],
                               self.size_x[mask],
                               self.size_y[mask],
                               self.frame_index[mask]]),
                    schema=headers
                )

            case _:
                return pl.DataFrame(
                    np.vstack([getattr(self, f)[mask] for f in self._reset_mapping[10]]),
                    schema=headers
                )

    def get_state_machine_dataframe(self) -> pl.DataFrame:
        """
        State Machine dataframe::

            ┌──────────┬───────┬──────────┬──────────┬──────────────┬───────────┐
            │ elapsed  ┆ cycle ┆ newState ┆ oldState ┆ stateElapsed ┆ trialType │
            │ ---      ┆ ---   ┆ ---      ┆ ---      ┆ ---          ┆ ---       │
            │ f64      ┆ f64   ┆ f64      ┆ f64      ┆ f64          ┆ f64       │
            ╞══════════╪═══════╪══════════╪══════════╪══════════════╪═══════════╡
            │ 902.601  ┆ 0.0   ┆ 1.0      ┆ 0.0      ┆ 902.601      ┆ 0.0       │
            │ 904.614  ┆ 0.0   ┆ 2.0      ┆ 1.0      ┆ 2.012        ┆ 0.0       │
            │ …        ┆ …     ┆ …        ┆ …        ┆ …            ┆ …         │
            │ 2711.425 ┆ 0.0   ┆ 3.0      ┆ 2.0      ┆ 3.01         ┆ 0.0       │
            │ 2711.425 ┆ 0.0   ┆ 0.0      ┆ 3.0      ┆ 0.0          ┆ 0.0       │
            └──────────┴───────┴──────────┴──────────┴──────────────┴───────────┘

        :return:
        """
        headers = self.log_header[20][1:]

        if self._reset_mapping is None:
            return pl.DataFrame(
                np.vstack([self.state_time,
                           self.state_cycle,
                           self.state_new_state,
                           self.state_old_state,
                           self.state_elapsed,
                           self.state_trial_type]),
                schema=headers
            )
        else:
            return pl.DataFrame(
                np.vstack([getattr(self, f) for f in self._reset_mapping[20]]),
                schema=headers
            )

    @property
    def exp_start_time(self) -> float:
        tstart = self.time[0]

        if isinstance(self.time_offset, float):
            return float(tstart + self.time_offset)
        elif isinstance(self.time_offset, np.ndarray):
            return float(tstart + self.time_offset[0])
        else:
            raise TypeError('')

    @property
    def exp_end_time(self) -> float:
        tend = self.time[-1]

        if isinstance(self.time_offset, float):
            return float(tend + self.time_offset)
        elif isinstance(self.time_offset, np.ndarray):
            return float(tend + self.time_offset[-1])
        else:
            raise TypeError('')

    @property
    def stim_start_time(self) -> float:
        v_start = np.nonzero(self.frame_index == 1)[0][0]
        tstart = self.time[v_start]

        if isinstance(self.time_offset, float):
            return float(tstart + self.time_offset)
        elif isinstance(self.time_offset, np.ndarray):
            return float(tstart + self.time_offset[0])
        else:
            raise TypeError('')

    @property
    def stim_end_time(self) -> float:
        v_end = np.nonzero(np.diff(self.frame_index) < 0)[0][-1] + 1
        tend = self.time[v_end]

        if isinstance(self.time_offset, float):
            return float(tend + self.time_offset)
        elif isinstance(self.time_offset, np.ndarray):
            return float(tend + self.time_offset[-1])
        else:
            raise TypeError('')

    @property
    def stimulus_segment(self) -> np.ndarray:
        v_start = self.frame_index == 1
        t1 = self.time[v_start]
        t2 = self.time[np.nonzero(np.diff(self.frame_index) < 0)[0] + 1]

        if isinstance(self.time_offset, float):
            offset = self.time_offset
        elif isinstance(self.time_offset, np.ndarray):
            offset = np.stack([self.time_offset, self.time_offset], axis=1)
        else:
            raise TypeError('')

        t = np.vstack((t1, t2)).T + offset

        return t

    @property
    def time_offset(self) -> float | np.ndarray:
        if self._cache_time_offset is None:
            self._cache_time_offset = diode_time_offset(
                self.riglog_data,
                self.diode_offset,
                return_sequential=self._do_sequential_diode_offset
            )
        return self._cache_time_offset

    @property
    def time_offset_statemachine(self) -> np.ndarray:
        """time offset approach using statemachine"""
        return self.state_time - self.s_on_v

    def session_trials(self) -> dict[Session, SessionInfo]:
        return {
            prot.name: prot
            for prot in get_protocol_sessions(self)
        }

    def get_stim_pattern(self, with_dur: bool = False) -> AbstractStimulusPattern:
        """
        Get stimulus pattern container

        :param with_dur: if extract theoretical duration value from protocol file
        :return: :class:`~stimpyp.base.AbstractStimulusPattern` based on stim type
        """
        prot = self.riglog_data.get_protocol()
        stim_type = self.riglog_data.get_stimulus_type()
        v_start = self.frame_index == 1
        t = self.stimulus_segment
        contrast = self.contrast[v_start]
        nr = self.stim_index[v_start]

        if stim_type == 'gratings':
            dire = self.ori[v_start]
            sf = self.sf[v_start]
            tf = prot.tf[nr]

            return GratingPattern(t, contrast, dire, sf, tf,
                                  duration=prot['dur'][nr] if with_dur else None)

        elif stim_type == 'functions':
            pos_x = self.pos_x[v_start]
            pos_y = self.pos_y[v_start]
            size_x = self.size_x[v_start]
            size_y = self.size_y[v_start]

            pos_xy = np.vstack([pos_x, pos_y]).T
            size_xy = np.vstack([size_x, size_y]).T

            return FunctionPattern(t, contrast, pos_xy, size_xy,
                                   duration=prot['dur'][nr] if with_dur else None)

        else:
            raise NotImplementedError('')

    @property
    def profile_dataframe(self) -> pl.DataFrame:
        x = np.logical_and(self.stim_index >= 0, self.trial_index != np.max(self.trial_index))  # TOOD check
        stim_index = self.stim_index[x]
        trial_index = self.trial_index[x]
        return pl.DataFrame({
            'i_stims': stim_index.astype(int),
            'i_trials': trial_index.astype(int)
        }).unique(maintain_order=True)


# ================= #
# Diode Time Offset #
# ================= #

class DiodeNumberMismatchError(ValueError):

    def __init__(self):
        super().__init__('Diode numbers are not detected reliably')


class DiodeSignalMissingError(RuntimeError):

    def __init__(self):
        super().__init__('no diode signal were found')


def diode_time_offset(rig: RiglogData,
                      diode_offset: bool = True,
                      return_sequential: bool = True,
                      default_offset_value: float = 0.6) -> float | np.ndarray:
    """
    time offset used in the `old stimpy`
    offset time from screen_time .riglog (diode) to .stimlog
    ** normally stimlog time value are smaller than riglog

    :param rig: :class:`RiglogData`
    :param diode_offset: whether correct diode signal
    :param return_sequential: return sequential offset, if False, use mean value across diode pulses
    :param default_offset_value: hardware(rig)-dependent offset value

    :return: tuple of offset average and std value
    """

    stimlog = rig.get_stimlog()
    if not isinstance(stimlog, Stimlog):
        raise TypeError('')

    if not diode_offset:
        warnings.warn('no offset')
        return default_offset_value

    #
    try:
        t = _diode_offset_sequential(rig, debug_plot=False)
    except DiodeNumberMismatchError as e:
        try:
            first_pulse = _check_if_diode_pulse(rig)
            logger.warning(f'{repr(e)}, use the first pulse diff for alignment')
            return first_pulse

        except DiodeSignalMissingError as e:
            logger.warning(f'{repr(e)}, use default value')
            return default_offset_value

    #
    avg_t = float(np.mean(t))
    std_t = float(np.std(t))

    if not (0 <= avg_t <= 1):
        logger.error(f'{avg_t} too large, might not be properly calculated, check...')

    logger.info(f'DIODE OFFSET avg: {avg_t}s, std: {std_t}s')

    if return_sequential:
        return t
    else:
        return avg_t


def _check_if_diode_pulse(rig: RiglogData) -> float:
    """only count 1st difference"""
    screen_time = rig.screen_event.time
    stimlog = rig.get_stimlog()
    if not isinstance(stimlog, Stimlog):
        raise TypeError('')
    stimlog_vstart = stimlog.time[(stimlog.frame_index == 1)]

    try:
        # noinspection PyTypeChecker
        return screen_time[0] - stimlog_vstart[0]
    except IndexError as e:
        raise DiodeSignalMissingError() from e


def _diode_offset_sequential(rig: RiglogData, debug_plot: bool = False) -> np.ndarray:
    stimlog = rig.get_stimlog()
    if not isinstance(stimlog, Stimlog):
        raise TypeError('')

    stimlog_vstart = stimlog.time[(stimlog.frame_index == 1)]
    riglog_vstart = rig.screen_event.time[0::2]

    if len(riglog_vstart) != len(stimlog_vstart):
        raise DiodeNumberMismatchError()
    else:
        if debug_plot:
            _plot_time_alignment_diode(riglog_vstart, stimlog_vstart)

        return riglog_vstart - stimlog_vstart


def _plot_time_alignment_diode(riglog_screen: np.ndarray,
                               stimlog_time: np.ndarray):
    """Plot time alignment (stimlog time value smaller than riglog)"""
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(riglog_screen - stimlog_time)
    ax.set(xlabel='Visual stim #', ylabel='Time diff (s)')


# ======== #
# Protocol #
# ======== #

def load_protocol(file: PathLike) -> StimpyProtocol:
    """Load stimpy protocol file

    :param file: protocol file path
    """
    return StimpyProtocol.load(file)


class StimpyProtocol(AbstractStimProtocol):
    r"""
    class for handle the protocol file for stimpy **bitbucket/github** version
    (mainly tested in the commits derived from master branch)

    `Dimension parameters`:

        N = numbers of visual stimulation (on-off pairs) = (T \* S)

        T = number of trials

        S = number of Stim Type

        C = number of Cycle
    """

    @classmethod
    def load(cls, file: PathLike) -> Self:
        file = Path(file)
        with file.open() as f:
            return cls._load(file.name, f)

    @classmethod
    def loads(cls, content: str, name: str = '<string>') -> Self:
        """load string"""
        return cls._load(name, content.split('\n'))

    @classmethod
    def _load(cls, name: str, content: Iterable[str]) -> Self:
        options = {}
        version = 'stimpy-bit'

        state = 0
        for line in content:
            line = line.strip()

            if len(line) == 0 or line.startswith('#'):
                continue

            # change state to 1 if # stimulus conditions
            if state == 0 and line.startswith('n '):
                header = re.split(' +', line)  # extract header
                data = [[] for _ in range(len(header))]
                state = 1

            elif state == 0:
                idx = line.index('=')
                value = try_casting_number(line[idx + 1:].strip())
                if isinstance(value, str) and '#' in value:
                    cmt = value.index('#')
                    value = value[:cmt].strip()
                options[line[:idx].strip()] = value

            elif state == 1:
                parts = re.split(' +', line, maxsplit=len(header))
                rows = unfold_stimuli_condition(parts)

                if len(rows) != 1:
                    version = 'stimpy-git'  # determine

                for r in rows:
                    for i, it in enumerate(r):  # for each col
                        data[i].append(it)
            else:
                raise RuntimeError('illegal state')

        assert len(header) == len(data)

        visual_stimuli = {
            field: list(map(str, data[i])) if field == 'evolveParams' else data[i]
            for i, field in enumerate(header)
        }

        # noinspection PyTypeChecker
        return StimpyProtocol(name, options, pl.DataFrame(visual_stimuli), version)

    @property
    def controller(self) -> str:
        """protocol controller name"""
        return self.options['controller']

    @property
    def is_shuffle(self) -> bool:
        return self.options['shuffle'] == 'True'

    @property
    def background(self) -> float:
        return float(self.options.get('background', 0.5))

    @property
    def start_blank_duration(self) -> int:
        return self.options.get('startBlankDuration', 5)

    @property
    def blank_duration(self) -> int:
        return self.options['blankDuration']

    @property
    def trial_blank_duration(self) -> int:
        return 2  # TODO not tested/checked value?

    @property
    def end_blank_duration(self) -> int:
        return self.options.get('endBlankDuration', 5)

    @property
    def trial_duration(self) -> int:
        dur = self['dur']
        return int(np.sum(dur) + len(dur) * self.blank_duration + self.trial_blank_duration)

    @property
    def visual_duration(self) -> int:
        return self.trial_duration * self.n_trials

    @property
    def total_duration(self) -> int:
        return self.start_blank_duration + self.visual_duration + self.end_blank_duration

    @property
    def texture(self) -> str:
        """stimulus texture, {circle, sqr... todo check in stimpy}"""
        return self.options['texture']

    @property
    def mask(self) -> str:
        """todo check in stimpy"""
        return self.options['mask']

    class EvoledParameter(dict[str, Any]):
        """handle the ``evolveParams`` header in the protocol file
        only available in **stimpy-bit** version
        """

        def __init__(self, keys: list[str], data: np.ndarray):
            self.__keys = keys
            self.__data = data  # {'phase':['linear',1]}

        def __len__(self) -> int:
            return len(self.__keys)

        def keys(self):
            return set(self.__keys)

        def __contains__(self, o) -> bool:
            return o in self.__keys

        def __getitem__(self, k: str) -> Any:
            if k not in self.__keys:  # i.e., 'phase'
                raise KeyError()

            # TODO only work for 'linear' case
            ret = []
            for d in self.__data:
                ret.append(d[k][1])

            return np.array(ret)

    def evolve_param_headers(self) -> list[str]:
        """Get parameter header which set in the 'evolveParams, i.g., 'phase'

        :return: header list.

        """
        keys = set()
        data = self['evolveParams']
        for d in data:
            d = eval(d)
            keys.update(d.keys())
        return list(keys)

    @property
    def evolve_params(self) -> EvoledParameter:
        """Get value from 'evolveParams'.

        Examples:

        >>> log: StimpyProtocol

        How many parameters in evolveParams

        >>> len(log.evolve_params)

        List parameter in 'evolveParams'

        >>> log.evolve_params.keys()

        Dose parameter 'phase' in 'evolveParams'?

        >>> 'phase' in log.evolve_params

        Get phase value from 'evolveParams'.

        >>> log.evolve_params['phase']

        :return:
        """
        if self.version == 'stimpy-git':
            raise DeprecationWarning('new stimpy has no evolveParams header')

        data = np.array([eval(it) for it in self['evolveParams']])  # cast back to dict
        return self.EvoledParameter(self.evolve_param_headers(), data)

    @property
    def tf(self) -> np.ndarray:
        if self.version == 'stimpy-bit':
            return self.evolve_params['phase']
        elif self.version == 'stimpy-git':
            return self['tf']
        else:
            raise NotImplementedError('')
