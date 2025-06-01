import dataclasses
import logging
import re
import warnings
from pathlib import Path
from typing import final, Any, Final

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from ._util import try_casting_number, unfold_stimuli_condition
from .base import AbstractLog, AbstractStimlog, AbstractStimProtocol
from .session import Session, SessionInfo
from .stimulus import GratingPattern

__all__ = ['PyVlog',
           'StimlogPyVStim',
           'PyVProtocol']

logger = logging.getLogger(__name__)


@final
class PyVlog(AbstractLog):
    """class for handle the log file (rig event specific) for pyvstim version (vb lab legacy)"""

    log_config: dict[str, Any] = {}
    log_info: dict[int, str] = {}
    log_header: dict[int, list[str]] = {}

    def __init__(self, *args, **kwargs):

        super().__init__(*args, log_suffix='.log', diode_offset=False, **kwargs)
        self.__prot_cache: PyVProtocol | None = None

    def _get_log_config(self) -> dict[str, Any]:
        """overwrite due to a single unify log"""
        with open(self.riglog_file) as f:
            for line in f:
                line = line.strip()

                if line.startswith('#'):

                    if 'Version' in line:
                        *_, val = line.split(' ')
                        self.log_config['version'] = val
                        logger.debug(f'Parsed version: {val}')

                    elif 'commit hash' in line:
                        heading, commit = line.split(': ')
                        self.log_config['commit_hash'] = commit
                        logger.debug(f'Parsed commit: {commit}')

                    elif 'CODES' in line:
                        heading, content = line.split(': ')
                        iter_codes = content.split(',')
                        for pair in iter_codes:
                            code, num = pair.split('=')
                            code = code.strip()
                            value = int(num.strip())
                            self.log_info[value] = code
                        logger.debug(f'Parsed log info: {self.log_info}')

                    elif 'VLOG HEADER' in line:
                        heading, content = line.split(':')
                        self.log_header[10] = content.split(',')
                        logger.debug(f'Parsed VLOG HEADER: {self.log_header[10]}')

                    elif 'RIG CSV' in line:
                        heading, content = line.split(': ')
                        info = self.log_info.copy()
                        info.pop(10)

                        for i in info.keys():
                            self.log_header[i] = content.split(',')

                        logger.debug(f'Parsed RIG CSV: {self.log_header}')

                    elif 'RIG VERSION' in line:
                        heading, content = line.split(': ')
                        self.log_config['rig_version'] = content
                        logger.debug(f'Parsed rig version: {self.log_config["rig_version"]}')

                    elif 'RIG GIT COMMIT HASH' in line:
                        heading, content = line.split(': ')
                        self.log_config['commit_hash'] = content
                        logger.debug(f'Parsed rig commit: {self.log_config["commit_hash"]}')

        self.log_config['source_version'] = 'pyvstim'

        return self.log_config

    @classmethod
    def _cache_asarray(cls, filepath: Path, square_brackets=False) -> np.ndarray:
        output = filepath.with_name(filepath.stem + '_log.npy')

        if not output.exists():

            data_list = []
            with filepath.open() as f:
                for line, content in enumerate(f):
                    content = content.strip()
                    if not content.startswith('#') and content != '':  # comments and empty line
                        cols = content.strip().split(',')
                        # Convert the columns to floats
                        cols = [float(x) for x in cols]
                        # Append the row to data_list
                        data_list.append(cols)

            # Find the maximum number of columns
            max_cols = max([len(row) for row in data_list])

            new_data = []

            # Iterate over each row
            for row in data_list:
                # Calculate the number of columns to add
                cols_to_add = max_cols - len(row)
                # Add the required number of np.nan values
                row.extend([np.nan] * cols_to_add)
                # Append the row to new_data
                new_data.append(row)

            # Convert new_data to a numpy array
            ret = np.array(new_data)

            np.save(output, ret)

        return np.load(output)

    # ===== #

    def get_stimlog(self) -> 'StimlogPyVStim':
        return StimlogPyVStim(self)

    def get_protocol(self) -> 'PyVProtocol':
        if self.__prot_cache is None:
            self.__prot_cache = PyVProtocol.load(self.prot_file)

        return self.__prot_cache


# ======= #
# Stimlog #
# ======= #

@final
class StimlogPyVStim(AbstractStimlog):
    """class for handle the log file (stim event specific) for pyvstim version (vb lab legacy)

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        P = number of acquisition sample pulse

    """

    frame: np.ndarray
    """TBD. value domain (-2, -1, 0, 1) ? Array[int, P] """

    blank: np.ndarray
    """If it is background only. 0: stim display, 1: no stim. Array[int, P]"""

    disp_x: np.ndarray
    """TBD. display pos x ? Array[int, P]"""

    disp_y: np.ndarray
    """TBD. display pos y ? Array[int, P]"""

    photo_state: np.ndarray
    """photo diode on-off. Array[int, P]. value domain in (0,1)"""

    v_duino_time: np.ndarray
    """extrapolate duinotime from screen indicator. sync arduino time in sec. Array[float, P]"""

    def __init__(self, riglog: 'PyVlog'):
        super().__init__(riglog, file_path=None)

        self.config: Final[dict[str, Any]] = riglog.log_config
        self.log_info: Final[dict[int, str]] = riglog.log_info
        self.log_header: Final[dict[int, list[str]]] = riglog.log_header

        self._reset()

    def _reset(self):
        self._reset_values()

    def _reset_values(self):
        _attrs = (
            'time',
            'stim_index',
            'trial_index',
            'frame',
            'blank',
            'contrast',
            'disp_x',
            'disp_y',
            'pos_x',
            'pos_y',
            'photo_state'
        )
        code = self.riglog_data.dat[:, 0] == 10
        for i, it in enumerate(self.riglog_data.dat[code, 1:].T):
            setattr(self, f'{_attrs[i]}', it)

        self.v_duino_time = self._get_stim_duino_time(self.riglog_data.dat[code, -1].T)

    def get_visual_stim_dataframe(self, **kwargs) -> pl.DataFrame:
        headers = self.log_header[10][1:]

        return pl.DataFrame(
            np.vstack([self.time / 1000,
                       self.stim_index,
                       self.trial_index,
                       self.frame,
                       self.blank,
                       self.contrast,
                       self.disp_x,
                       self.disp_y,
                       self.pos_x,
                       self.pos_y,
                       self.photo_state]),
            schema=headers
        )

    def get_state_machine_dataframe(self):
        raise RuntimeError('no info in pyvstim version')

    def _get_stim_duino_time(self, indicator_flag: np.ndarray) -> np.ndarray:
        """extrapolate duinotime from screen indicator. sync arduino time in (P,) sec"""
        fliploc = np.where(
            np.diff(np.hstack([0, indicator_flag, 0])) != 0
        )[0]

        return interp1d(
            fliploc,
            self.riglog_data.screen_event.time,
            fill_value="extrapolate"
        )(np.arange(len(indicator_flag)))

    @property
    def exp_start_time(self) -> float:
        return float(self.v_duino_time[0])

    @property
    def stimulus_segment(self) -> np.ndarray:
        ustims = self.stim_index * (1 - self.blank)
        utrials = self.trial_index * (1 - self.blank)

        n_trials = self.profile_dataframe.shape[0]
        ret = np.zeros([n_trials, 2])
        for i, (st, tr) in enumerate(self.profile_dataframe.iter_rows()):
            idx = np.where((ustims == st) & (utrials == tr))[0]
            ret[i, :] = self.v_duino_time[[idx[0], idx[-1]]]

        return ret

    def session_trials(self) -> dict[Session, SessionInfo]:
        raise NotImplementedError('')

    @property
    def time_offset(self) -> float:
        """directly used interpolation using diode signal already"""
        return 0

    def get_stim_pattern(self) -> GratingPattern:
        raise NotImplementedError('')

    def exp_end_time(self) -> float:
        return float(self.v_duino_time[-1])

    # =========== #
    # retinotopic #
    # =========== #

    @property
    def stim_loc(self) -> np.ndarray:
        return np.vstack([self.pos_x, self.pos_y]).T

    @property
    def avg_refresh_rate(self) -> float:
        """in Hz"""
        return 1 / (np.diff(self.v_duino_time).mean())

    def plot_stim_animation(self):
        from ._util import plot_scatter_animation
        plot_scatter_animation(self.pos_x,
                               self.pos_y,
                               self.v_duino_time,
                               step=int(self.avg_refresh_rate))  # TODO check refresh rate?

    @property
    def profile_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({
            'i_stims': self.stim_index.astype(int),
            'i_trials': self.trial_index.astype(int)
        }).unique(maintain_order=True)

    @property
    def n_cycles(self) -> list[int]:
        return self.riglog_data.get_protocol().get_loops_expr().n_cycles


# ======== #
# Protocol #
# ======== #

class PyVProtocol(AbstractStimProtocol):
    r"""
    class for handle the protocol file for pyvstim version (vb lab legacy)

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T \* S)

        T = number of trials

        S = number of Stim Type

        C = number of Cycle
    """

    @classmethod
    def load(cls, file: Path | str, *,
             cast_numerical_opt=True) -> 'PyVProtocol':

        file = Path(file)
        options = {}
        version = 'pyvstim'

        state = 0
        with Path(file).open() as f:
            for line in f:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                if state == 0 and line.startswith('n\t'):
                    header = re.split(r'\t+| +', line)
                    value = [[] for _ in range(len(header))]
                    state = 1

                elif state == 0:
                    idx = line.index('=')
                    if cast_numerical_opt:
                        opt_value = try_casting_number(line[idx + 1:].strip())
                    else:
                        opt_value = line[idx + 1:].strip()

                    options[line[:idx].strip()] = opt_value

                elif state == 1:
                    parts = re.split(r'\t+| +', line, maxsplit=len(header))
                    rows = unfold_stimuli_condition(parts)

                    for r in rows:
                        r.remove('')  # pyvstim interesting problem
                        for i, it in enumerate(r):  # for each col
                            if it != '':
                                value[i].append(it)
                else:
                    raise RuntimeError('illegal state')

            assert len(header) == len(value)
            visual_stimuli = {
                field: value[i]
                for i, field in enumerate(header)
            }

            if 'Shuffle' not in options.keys():
                options['Shuffle'] = False

        return PyVProtocol(file.name, options, pl.DataFrame(visual_stimuli), version)

    @property
    def is_shuffle(self) -> bool:
        """TODO"""
        return False

    @property
    def background(self) -> float:
        """TODO"""
        return self.options.get('background', 0.5)

    @property
    def start_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def blank_duration(self) -> int:
        return self.options['BlankDuration']

    @property
    def trial_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def end_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def trial_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def visual_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def total_duration(self) -> int:
        raise NotImplementedError('')

    def get_loops_expr(self) -> 'ProtExpression':
        """parse and get the expression and loop number"""
        exprs = []
        n_cycles = []
        n_blocks = self.visual_stimuli_dataframe.shape[0]

        for row in self.visual_stimuli_dataframe.iter_rows():  # each row item_value
            for it in row:
                if isinstance(it, str):
                    if 'loop' in it:
                        match = re.search(r"loop\((.*),(\d+)\)", it)

                        if match:
                            exprs.append(match.group(1))
                            n_cycles.append(match.group(2))
                    else:
                        warnings.warn('loop info not found, check prot file!', vtype='warning')
                        exprs.append('')
                        n_cycles.append(1)

        return ProtExpression(exprs, list(map(int, n_cycles)), n_blocks)


@dataclasses.dataclass
class ProtExpression:
    """
    `Dimension parameters`:

        B = number of block

        C = number of Cycle
    """

    expr: list[str]
    """expression"""
    n_cycles: list[int]
    """number of cycle. length number = B?, value equal to C"""
    n_blocks: int | None
    """number of prot value row (block) B"""

    def __post_init__(self):
        if (len(self.n_cycles) == 2 * self.n_blocks) and self._check_ncycles_foreach_block():
            self.n_cycles = self.n_cycles[::2]
        else:
            raise RuntimeError('')

    def _check_ncycles_foreach_block(self):
        """check if the ncycles are the same and duplicate for each block"""
        n = len(self.n_cycles)
        if n % 2 != 0:
            return False

        for i in range(0, n, 2):
            if self.n_cycles[i] != self.n_cycles[i + 1]:
                return False

        return True
