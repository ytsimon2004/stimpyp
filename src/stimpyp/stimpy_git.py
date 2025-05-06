import re
from pathlib import Path
from typing import Any, Callable, final

import numpy as np
import polars as pl

from ._type import PathLike
from ._util import deprecated_func
from .base import AbstractStimlog
from .session import Session, SessionInfo, get_protocol_sessions
from .stimpy_core import RiglogData, StimpyProtocol
from .stimulus import GratingPattern

__all__ = ['StimlogGit',
           'lazy_load_stimlog']


@final
class StimlogGit(AbstractStimlog):
    """class for handle the stimlog file for stimpy **github** version
    (mainly tested in the commits derived from master branch)

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        I = number of photo indicator pulse
    """

    log_info: dict[int, str]
    """{0: <'Gratings', ...>, 1: 'PhotoIndicator', 2: 'StateMachine', 3: 'LogDict'}"""
    log_header: dict[int, list[str]]
    """list of header"""
    log_data: dict[int, list[tuple[Any, ...]]]
    """only for StateMachine(2) and LogDict(3)"""

    # =========== #
    # Visual Pars #
    # =========== #

    duration: np.ndarray
    """duration in sec. Array[float, P]"""
    pos_xy: np.ndarray
    """object center position XY. Array[float, [P, 2]]"""
    size_xy: np.ndarray
    """object size width and height. Array[int, [P, 2]]"""

    # ============== #
    # PhotoIndicator #
    # ============== #

    photo_time: np.ndarray
    """photoindicator time. Array[float, I]"""
    photo_state: np.ndarray
    """photoindicator state. Array[bool, I]"""
    photo_size: np.ndarray
    """photoindicator size in given unit. Array[float, I]"""
    photo_pos: np.ndarray
    """photoindicator in XY. Array[int, [I, 2]]"""
    photo_units: np.ndarray
    """photoindicator unit. Array[str, I]"""
    photo_mode: np.ndarray
    """photoindicator mode. Array[int, I]"""
    photo_frames: np.ndarray
    """photoindicator frames . Array[int, I]"""
    photo_enable: np.ndarray
    """photoindicator size in pixel. Array[bool, I]"""

    def __init__(self, riglog: RiglogData,
                 file_path: PathLike,
                 reset_mapping: dict[int, list[str]] | None = None,
                 diode_offset: bool = True):

        super().__init__(riglog, file_path, reset_mapping)
        self.diode_offset = diode_offset
        self._cache_time_offset: float | None = None

        self._reset()

    def _reset(self):
        log_data = {}

        with self.stimlog_file.open() as f:
            for line, content in enumerate(f):  # type: int, str
                content = content.strip()
                # `## (CODE VERSION) : (commit hash: 88c4705 - tags: [''])`
                # *: 0-inf; +: 1-inf; ?: 0-1
                m = re.match(r'#+ (.+?)\s*:\s*(.+)', content)
                if m:
                    info_name = m.group(1)
                    info_value = m.group(2)

                    if info_name == 'LOG NAME':
                        self.config['log_name'] = info_value
                    elif info_name == 'CODE VERSION':
                        self._reset_code_version(info_value)
                    elif info_name == 'Format':
                        self.config['format'] = info_value.split(' ')
                        if self.config['format'] != ['source_id', 'time', 'source_infos']:
                            raise RuntimeError('stimlog format changed')
                    elif info_name == 'Rig trigger on':
                        info_value = info_value.split(',')
                        self.config['rig_trigger'] = (info_value[0], float(info_value[1]))
                    elif self._reset_log_info(info_name, info_value):
                        pass
                    else:
                        print(f'ignore header line at {line + 1} : {content}')

                elif content.startswith('### START'):
                    self.config['start_time'] = float(content[9:].strip())
                elif content.startswith('### END'):
                    self.config['end_time'] = float(content[7:].strip())
                elif content.startswith('# Missed') and content.endswith('frames'):
                    self.config['missed_frames'] = int(content.split(' ')[2])
                elif content.startswith('#'):
                    print(f'ignore header line at {line + 1} : {content}')

                else:
                    try:
                        self._reset_line(log_data, line + 1, content)
                    except BaseException:
                        raise RuntimeError(f'bad format line at {line + 1} : {content}')

        self.log_data = self._reset_final(log_data)

    def _reset_code_version(self, info_value: str):
        """for ### CODE VERSION,  commit hash and tags"""
        info_value = info_value.strip().split(' ')
        commits = info_value[info_value.index('hash:') + 1]
        tag = info_value[info_value.index('tags:') + 1]

        self.config['commit_hash'] = commits if commits != 'None' else None
        self.config['tag'] = tag if tag != 'None' else None

    def _reset_log_info(self, code: str, info_value: str):
        try:
            code = int(code)
        except ValueError:
            return False

        name, header = info_value.split(' ', maxsplit=1)
        header = eval(header)
        self.log_info[code] = name
        self.log_header[code] = header
        return True

    def _reset_line(self, log_data: dict[int, list], line: int, content: str):
        message: str
        value: list
        code, time, message = content.split(' ', maxsplit=2)
        code = int(code)
        time = float(time)

        if self.log_info[code] in ('Gratings', 'FunctionBased', 'PhotoIndicator', 'LogDict'):
            value = eval(message)
            if len(self.log_header[code]) != len(value):
                print(f'log category {code} size mismatched at line {line} : {message}')
            else:
                log_data.setdefault(code, []).append((time, *value))

        elif self.log_info[code] == 'StateMachine':
            value = eval(message.replace('<', '("').replace(':', '",').replace('>', ')'))
            value = list(map(str, value))
            if len(self.log_header[code]) != len(value):
                print(f'log category {code} size mismatched at line {line} : {message}')
            else:
                log_data.setdefault(code, []).append((time, *value))

        else:
            print(f'unknown log category : at line {line} : {content}')

    def _reset_final(self, log_data: dict[int, list]) -> dict[int, list]:
        remove_code = []
        for code, content in log_data.items():
            if self.log_info[code] in ('Gratings', 'FunctionBased'):
                self.time = np.array([it[0] for it in content])
                self.duration = np.array([it[1] for it in content])
                self.contrast = np.array([it[2] for it in content])
                self.ori = np.array([it[3] for it in content])
                self.phase = np.array([it[4] for it in content])
                self.pos_xy = np.array([it[5] for it in content])
                self.size_xy = np.array([it[6] for it in content])
                self.flick = np.array([it[7] for it in content])
                self.interpolate = np.array([it[8] for it in content], dtype=bool)
                self.mask = np.array([it[9] is not None for it in content], dtype=bool)
                self.sf = np.array([it[10] for it in content])
                self.tf = np.array([it[11] for it in content])
                self.opto = np.array([it[12] for it in content], dtype=int)
                self.pattern = np.array([it[13] for it in content])

                remove_code.append(code)

            elif self.log_info[code] == 'PhotoIndicator':
                self.photo_time = np.array([it[0] for it in content])
                self.photo_state = np.array([it[1] for it in content], dtype=bool)
                self.photo_size = np.array([it[2] for it in content])
                self.photo_pos = np.array([it[3] for it in content])
                self.photo_units = np.array([it[4] for it in content])
                self.photo_mode = np.array([it[5] for it in content], dtype=int)
                self.photo_frames = np.array([it[6] for it in content], dtype=int)
                self.photo_enable = np.array([it[7] for it in content], dtype=bool)

                remove_code.append(code)

        for code in remove_code:
            del log_data[code]

        return log_data

    def get_visual_stim_dataframe(self) -> pl.DataFrame:
        """
        Visual presentation dataframe::

            ┌────────────┬──────────┬──────────┬─────┬───┬──────┬─────┬──────┬─────────┐
            │ time       ┆ duration ┆ contrast ┆ ori ┆ … ┆ sf   ┆ tf  ┆ opto ┆ pattern │
            │ ---        ┆ ---      ┆ ---      ┆ --- ┆   ┆ ---  ┆ --- ┆ ---  ┆ ---     │
            │ f64        ┆ i64      ┆ i64      ┆ i64 ┆   ┆ f64  ┆ i64 ┆ i64  ┆ str     │
            ╞════════════╪══════════╪══════════╪═════╪═══╪══════╪═════╪══════╪═════════╡
            │ 18.990026  ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 10  ┆ 0    ┆ square  │
            │ 21.000029  ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 10  ┆ 0    ┆ square  │
            │ …          ┆ …        ┆ …        ┆ …   ┆ … ┆ …    ┆ …   ┆ …    ┆ …       │
            │ 619.054972 ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 50  ┆ 0    ┆ square  │
            │ 619.084972 ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 50  ┆ 0    ┆ square  │
            └────────────┴──────────┴──────────┴─────┴───┴──────┴─────┴──────┴─────────┘

        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(self.time).alias('time'),
            pl.Series(self.duration).alias('duration'),
            pl.Series(self.contrast).alias('contrast'),
            pl.Series(self.ori).alias('ori'),
            pl.Series(self.phase).alias('phase'),
            pl.Series(self.pos_xy).alias('pos'),
            pl.Series(self.size_xy).alias('size'),
            pl.Series(self.flick).alias('flick'),
            pl.Series(self.interpolate).alias('interpolate'),
            pl.Series(self.mask).alias('mask'),
            pl.Series(self.sf).alias('sf'),
            pl.Series(self.tf).alias('tf'),
            pl.Series(self.opto).alias('opto'),
            pl.Series(self.pattern).alias('pattern')
        )
        return df

    def get_photo_indicator_dataframe(self) -> pl.DataFrame:
        """
        PhotoDiode dataframe::

            ┌────────────┬───────┬──────┬────────────┬───────┬──────┬────────┬────────┐
            │ time       ┆ state ┆ size ┆ pos        ┆ units ┆ mode ┆ frames ┆ enable │
            │ ---        ┆ ---   ┆ ---  ┆ ---        ┆ ---   ┆ ---  ┆ ---    ┆ ---    │
            │ f64        ┆ bool  ┆ i64  ┆ list[i64]  ┆ str   ┆ i64  ┆ i64    ┆ bool   │
            ╞════════════╪═══════╪══════╪════════════╪═══════╪══════╪════════╪════════╡
            │ 18.990026  ┆ false ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
            │ 21.000029  ┆ true  ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
            │ …          ┆ …     ┆ …    ┆ …          ┆ …     ┆ …    ┆ …      ┆ …      │
            │ 607.094955 ┆ false ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
            │ 609.104958 ┆ true  ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
            └────────────┴───────┴──────┴────────────┴───────┴──────┴────────┴────────┘

        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(self.photo_time).alias('time'),
            pl.Series(self.photo_state).alias('state'),
            pl.Series(self.photo_size).alias('size'),
            pl.Series(self.photo_pos).alias('pos'),
            pl.Series(self.photo_units).alias('units'),
            pl.Series(self.photo_mode).alias('mode'),
            pl.Series(self.photo_frames).alias('frames'),
            pl.Series(self.photo_enable).alias('enable')
        )
        return df

    def get_state_machine_dataframe(self) -> pl.DataFrame:
        """
        State Machine dataframe::

            ┌────────────┬───────────────────────────┬───────────────────────────┐
            │ time       ┆ state                     ┆ prev_state                │
            │ ---        ┆ ---                       ┆ ---                       │
            │ f64        ┆ str                       ┆ str                       │
            ╞════════════╪═══════════════════════════╪═══════════════════════════╡
            │ 18.990026  ┆ ('States.SHOW_BLANK', 1)  ┆ ('States.STIM_SELECT', 0) │
            │ 20.990029  ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
            │ …          ┆ …                         ┆ …                         │
            │ 609.094958 ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
            │ 619.094972 ┆ ('States.STIM_SELECT', 0) ┆ ('States.SHOW_STIM', 2)   │
            └────────────┴───────────────────────────┴───────────────────────────┘

        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(it[0] for it in self.log_data[2]).alias('time'),
            pl.Series(it[1] for it in self.log_data[2]).alias('state'),
            pl.Series(it[2] for it in self.log_data[2]).alias('prev_state'),
        )
        return df

    def get_log_dict_dataframe(self) -> pl.DataFrame:
        """
        Log Dict DataFrame::

            ┌────────────┬──────────┬──────────┬──────────────┬────────────┐
            │ time       ┆ block_nr ┆ trial_nr ┆ condition_nr ┆ trial_type │
            │ ---        ┆ ---      ┆ ---      ┆ ---          ┆ ---        │
            │ f64        ┆ i64      ┆ i64      ┆ i64          ┆ i64        │
            ╞════════════╪══════════╪══════════╪══════════════╪════════════╡
            │ 18.990026  ┆ 0        ┆ 0        ┆ 0            ┆ 1          │
            │ 30.990043  ┆ 1        ┆ 0        ┆ 0            ┆ 1          │
            │ …          ┆ …        ┆ …        ┆ …            ┆ …          │
            │ 595.083939 ┆ 48       ┆ 0        ┆ 0            ┆ 1          │
            │ 607.094955 ┆ 49       ┆ 0        ┆ 0            ┆ 1          │
            └────────────┴──────────┴──────────┴──────────────┴────────────┘

        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(it[0] for it in self.log_data[3]).alias('time'),
            pl.Series(it[1] for it in self.log_data[3]).alias('block_nr'),
            pl.Series(it[2] for it in self.log_data[3]).alias('trial_nr'),
            pl.Series(it[3] for it in self.log_data[3]).alias('condition_nr'),
            pl.Series(it[4] for it in self.log_data[3]).alias('trial_type'),
        )

        return df

    def get_column(self,
                   category: int | str,
                   field: str,
                   dtype=float,
                   mapper: Callable[[Any], Any] | None = None) -> np.ndarray:

        if category in ('Gratings', 'FunctionBased', 'PhotoIndicator'):
            raise RuntimeError(f'preprocessed log : {category}:{field}')

        if isinstance(category, str):
            category = self._get_category_code(category)

        field = self._get_field_index(category, field)
        data = [it[field] for it in self.log_data[category]]

        if mapper is not None:
            # i.e., operator.itemgetter(1) for tuple statemachine ('States.SHOW_STIM', 2)...
            data = list(map(mapper, data))

        return np.array(data, dtype=dtype)

    def _get_category_code(self, category: str) -> int:
        for code, name in self.log_info.items():
            if name == category:
                return code
        raise KeyError('')

    def _get_field_index(self, category: int, field: str) -> int:
        header = self.log_header[category]
        if field == 'time':
            return 0
        return header.index(field) + 1

    @property
    def pos_x(self) -> np.ndarray:
        """object center position X. Array[float, P]"""
        return self.pos_xy[:, 0]

    @property
    def pos_y(self) -> np.ndarray:
        """object center position Y. Array[float, P]"""
        return self.pos_xy[:, 1]

    @property
    def size_x(self) -> np.ndarray:
        """object size width. Array[int, P]"""
        return self.size_xy[:, 0]

    @property
    def size_y(self) -> np.ndarray:
        """object size height. Array[int, P]"""
        return self.size_xy[:, 1]

    @property
    def exp_start_time(self) -> float:
        return self.riglog_data.dat[0, 2] / 1000

    @property
    def exp_end_time(self) -> float:
        return self.riglog_data.dat[-1, 2] / 1000

    @property
    def stim_start_time(self) -> float:
        return float(self.photo_time[1] + self.time_offset)

    @property
    def stim_end_time(self) -> float:
        return float(self.time[-1] + self.time_offset)

    @property
    def stimulus_segment(self) -> np.ndarray:
        # ret = self.riglog_data.screen_event[:, 0].reshape(-1, 2)  # directly use riglog time
        # stimlog bug starting from `diode off`, and ending from `diode True`
        # thus remove first point and add last point manually
        _p_time = np.concatenate((self.photo_time[1:], np.array([self.time[-1]])))
        ret = _p_time.reshape(-1, 2) + self.time_offset
        return ret

    def session_trials(self) -> dict[Session, SessionInfo]:
        return {
            prot.name: prot
            for prot in get_protocol_sessions(self)
        }

    @property
    def time_offset(self) -> float:
        if self._cache_time_offset is None:
            # noinspection PyTypeChecker
            self._cache_time_offset = _time_offset(self.riglog_data, self, self.diode_offset)[0]
        return self._cache_time_offset

    def get_stim_pattern(self) -> GratingPattern:
        prot = StimpyProtocol.load(self.stimlog_file.with_suffix('.prot'))
        log_nr = self.get_column('LogDict', 'condition_nr').astype(int)

        t = self.stimulus_segment
        ori = np.array([prot['ori'][n] for n in log_nr])
        sf = np.array([prot['sf'][n] for n in log_nr])
        tf = np.array([prot['tf'][n] for n in log_nr])
        contrast = np.array([prot['c'][n] for n in log_nr])
        dur = np.array([prot['dur'][n] for n in log_nr])

        return GratingPattern(t, ori, sf, tf, contrast, duration=dur)

    @property
    def profile_dataframe(self) -> pl.DataFrame:
        return (
            self.get_log_dict_dataframe()
            .with_columns(pl.col('condition_nr').alias('i_stims'))
            .with_columns(pl.col('block_nr').alias('i_trials'))
            .select('i_stims', 'i_trials')
        )


# ========== #

@deprecated_func(remarks='generalized, to sequential offset method')
def _time_offset(rig: RiglogData,
                 stm: StimlogGit,
                 diode_offset=True) -> tuple[float, float]:
    """
    time offset used in the `new stimpy`
    offset time from screen_time .riglog (diode) to .stimlog
    ** normally stimlog time value are larger than riglog

    :param rig:
    :param stm:
    :param diode_offset: whether correct diode signal
    :return: tuple of offset average and std value
    """
    if not isinstance(stm, StimlogGit):
        raise TypeError('')

    screen_time = rig.screen_event.time

    if diode_offset:
        try:
            # new stimpy might be a negative value (stimlog time value larger than riglog)
            # stimlog bug starting from `diode off`, and ending from `diode True`
            # thus remove first point and add last point manually
            _p_time = np.concatenate((stm.photo_time[1:], np.array([stm.time[-1]])))
            offset_t = screen_time[::2] - _p_time[::2]

        except ValueError as e:
            print(f'number of diode pulse and stimulus mismatch from {e}')
            print(f'use the first pulse diff for alignment')
            offset_t = screen_time[0] - stm.photo_time[1]
    else:
        raise NotImplementedError('')

    offset_t_avg = float(np.mean(offset_t))
    offset_t_std = float(np.std(offset_t))

    print(f'time offset between stimlog and riglog: {round(offset_t_avg, 3)}')
    print(f'offset_std: {round(offset_t_std, 3)}')

    return offset_t_avg, offset_t_std


def lazy_load_stimlog(file: PathLike, string_key: bool = True) -> dict[str | int, pl.DataFrame]:
    """
    Load directly the stimlog file (without riglog time offset), and parse data as polars dataframes

    .. code-block:: python

        file = ...

        log = load_stimlog(file, string_key=True)  # get dataframe using keyname
        print(log['PhotoIndicator'])

        log = load_stimlog(file, string_key=False)  # get dataframe using code int
        print(log[1])

    :param file: file path for the .stimlog
    :param string_key: show key as str type, otherwise, int type
    :return: Code:DataFrame dictionary
    """

    log_info = {}
    log_header = {}
    log_data = {}

    with Path(file).open() as f:
        for line, content in enumerate(f):
            content = content.strip()

            #
            m = re.match(r'#+ (.+?)\s*:\s*(.+)', content)
            if m:
                info_name = m.group(1)
                info_value = m.group(2)
                try:
                    code = int(info_name)
                except ValueError:
                    continue
                else:
                    name, header = info_value.split(' ', maxsplit=1)
                    header = eval(header)
                    log_info[code] = name
                    log_header[code] = header

            elif content.startswith('#'):
                print(f'ignore header line at {line + 1} : {content}')
                continue

            else:  # data
                code, time, message = content.split(' ', maxsplit=2)
                code = int(code)
                time = float(time)

                if log_info[code] == 'StateMachine':
                    value = eval(message.replace('<', '("').replace(':', '",').replace('>', ')'))
                    value = list(map(str, value))
                    if len(log_header[code]) != len(value):
                        print(f'log category {code} size mismatched at line {line} : {message}')
                    else:
                        log_data.setdefault(code, []).append((time, *value))

                else:
                    value = eval(message)
                    if len(log_header[code]) != len(value):
                        print(f'log category {code} size mismatched at line {line} : {message}')
                    else:
                        log_data.setdefault(code, []).append((time, *value))

        return {
            (log_info[code] if string_key else code):
                pl.DataFrame(log_data[code], schema=['time'] + log_header[code], strict=False)
            for code in log_data
        }
