from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Literal, TypeVar, Generic, TypedDict, Any, overload, Iterable

import numpy as np
import polars as pl
from polars.polars import ColumnNotFoundError
from typing_extensions import Self

from ._type import PathLike
from ._util import printdf
from .event import RigEvent, CamEvent
from .preference import PreferenceDict, load_preferences
from .session import Session, SessionInfo

__all__ = [
    'STIMPY_SOURCE_VERSION',
    'LOG_SUFFIX',
    'CAMERA_TYPE',
    #
    'RigConfig',
    'AbstractLog',
    'AbstractStimlog',
    #
    'AbstractStimProtocol',
    #
    'AbstractStimulusPattern'
]

STIMPY_SOURCE_VERSION = Literal['pyvstim', 'stimpy-bit', 'stimpy-git', 'debug']
LOG_SUFFIX = Literal['.log', '.riglog']
CAMERA_TYPE = Literal['facecam', 'eyecam', '1P_cam']

#
R = TypeVar('R', bound='AbstractLog')  # Riglog-Like
S = TypeVar('S', bound='AbstractStimlog')  # Stimlog-Like
P = TypeVar('P', bound='AbstractStimProtocol')  # protocol file

#
logger = logging.getLogger(__name__)


# =========== #
# Baselog ABC #
# =========== #

class RigConfig(TypedDict, total=False):
    source_version: STIMPY_SOURCE_VERSION
    """stimpy source version {'pyvstim', 'stimpy-bit', 'stimpy-git'}"""
    version: float
    """acquisition flag. i.e., 0.3"""
    commit_hash: str
    """git commit hash"""
    codes: dict[str, int]
    """<EVENT_TYPES>:<NUMBER>"""
    fields: tuple[str, ...]
    """column repr for the logging"""


class AbstractLog(Generic[S, P], metaclass=abc.ABCMeta):
    """ABC class for different stimpy/pyvstim log files. i.e., .log, .riglog"""

    log_config: RigConfig
    """config dict for the log file"""

    def __init__(self, root_path: PathLike,
                 log_suffix: LOG_SUFFIX,
                 diode_offset: bool = True,
                 reset_mapping: dict[int, list[str]] | None = None):
        """
        :param root_path: log file path or log directory
        :param log_suffix: log file suffix
        :param diode_offset: whether do the diode offset
        :param reset_mapping: Customized mapping
        """

        if not isinstance(root_path, Path):
            root_path = Path(root_path)

        if root_path.is_dir():
            self.riglog_file = self._find_logfile(root_path, log_suffix)
        else:
            self.riglog_file = root_path

        # set when get config
        self._with_square_brackets: bool = True
        self._skip_rows: int = 0

        self.log_config = self._get_log_config()
        self.version = self.log_config['source_version']
        self.dat = self._cache_asarray(self.riglog_file, self._with_square_brackets)

        #
        self._diode_offset = diode_offset
        self._reset_mapping = reset_mapping

    @classmethod
    def _find_logfile(cls, root: Path,
                      log_suffix: LOG_SUFFIX) -> Path:

        f = list(root.glob(f'*{log_suffix}'))
        if len(f) == 1:
            return f[0]

        elif len(f) == 0:
            logger.warning(f'no riglog under {root}, try to find in the subfolder...')
            for s in root.iterdir():
                if s.is_dir() and s.name.startswith('run0'):
                    try:
                        return cls._find_logfile(s, log_suffix)
                    except FileNotFoundError:
                        pass

            raise FileNotFoundError(f'no riglog file {log_suffix} under {root}')

        else:
            raise FileNotFoundError(f'more than one riglog files under {root}')

    # noinspection PyTypedDict
    def _get_log_config(self) -> RigConfig:
        """get config dict and source version for different logs from the # headers"""
        ret = RigConfig()
        with open(self.riglog_file) as f:
            for line in f:
                if '#' in line:
                    if 'RIG VERSION' in line:
                        ret['version'] = float(line.split(': ')[-1])
                        logger.debug(f'Parsed version: {ret["version"]}')

                    elif 'RIG GIT COMMIT HASH' in line:
                        ret['commit_hash'] = line.split(': ')[-1].strip()
                        logger.debug(f'Parsed commit_hash: {ret["commit_hash"]}')

                    elif 'CODES' in line:
                        codes = {}
                        content = line.replace('# CODES: ', '').strip()
                        iter_codes = content.split(',')
                        for pair in iter_codes:
                            code, num = pair.split('=')
                            code = code.strip()
                            value = int(num.strip())
                            codes[code.lower()] = value

                        ret['codes'] = codes
                        logger.debug(f'Parsed codes: {ret["codes"]}')

                    elif 'RIG CSV' in line:
                        content = line.replace('# RIG CSV: ', '').strip()
                        ret['fields'] = tuple(content.split(','))
                        logger.debug(f'Parsed fields: {ret["fields"]}')

                # 2025 new update log...
                elif '#' not in line and 'code' in line:
                    ret['fields'] = tuple(line.split(','))
                    self._skip_rows = 1
                    self._with_square_brackets = False
                    logger.debug(f'github stimpy version parsed: fields: {ret["fields"]}, without square brackets')

        # infer
        if 'opto' not in ret['codes']:
            ret['source_version'] = 'stimpy-bit'
            logger.info(f'Source version inferred by *opto* field: {ret["source_version"]}')

        if 'version' not in ret:
            ret['source_version'] = 'stimpy-git'
            logger.info(f'Source version inferred by config: {ret["source_version"]}')

        if 'opto' in ret['codes'] and ret['codes']['opto'] == 15:
            ret['source_version'] = 'pyvstim'
            logger.info(f'Source version inferred by *opto* field: {ret["source_version"]}')

        return ret

    @classmethod
    @abc.abstractmethod
    def _cache_asarray(cls, filepath: Path, square_brackets: bool) -> np.ndarray:
        pass

    # ============ #
    # RIGLOG EVENT #
    # ============ #

    def _event(self, code: int) -> np.ndarray:
        """shape(sample, 2) with time and value"""
        x = self.dat[:, 0] == code
        t = self.dat[x, 2].copy() / 1000  # s
        v = self.dat[x, 3].copy()
        return np.vstack([t, v]).T

    @property
    def exp_start_time(self) -> float:
        """experimental start time (in sec)"""
        return float(self.dat[0, 2].copy() / 1000)

    @property
    def exp_end_time(self) -> float:
        """experimental end time (in sec)"""
        return float(self.dat[-1, 2].copy() / 1000)

    @property
    def total_duration(self) -> float:
        """experimental duration (in sec)"""
        return self.exp_end_time - self.exp_start_time

    @property
    def screen_event(self) -> RigEvent:
        """screen rig event. i.e., diode pulse"""
        return RigEvent('screen', self._event(0))

    @property
    def imaging_event(self) -> RigEvent:
        """imaging rig event. i.e., 2photon pulse"""
        return RigEvent('imaging', self._event(1))

    @property
    def position_event(self) -> RigEvent:
        """position rig event. i.e., encoder pulse"""
        return RigEvent('position', self._event(2))

    @property
    def lick_event(self) -> RigEvent:
        """lick rig event. i.e., lick meter pulse"""
        return RigEvent('lick', self._event(3))

    @property
    def reward_event(self) -> RigEvent:
        """reward rig event. i.e., reward given pulse from lick meter"""
        if self.version == 'stimpy-git':
            return RigEvent('reward', self._event(5))
        else:
            return RigEvent('reward', self._event(4))

    @property
    def lap_event(self) -> RigEvent:
        """lap rig event. i.e., optic sensing for the reflective taps"""
        if self.version == 'stimpy-git':
            return RigEvent('lap', self._event(6))
        else:
            return RigEvent('lap', self._event(5))

    @property
    def act_event(self) -> RigEvent:
        """todo"""
        raise NotImplementedError('')

    @property
    def opto_event(self) -> RigEvent:
        """todo"""
        raise NotImplementedError('')

    class CameraEvent:
        """camera event"""
        camera: dict[CAMERA_TYPE, int]

        def __init__(self, rig: R):
            """
            :param rig:``Baselog``
            """
            self.rig = rig

            if rig.version == 'stimpy-git':
                self.camera = {
                    'facecam': 7,
                    'eyecam': 8,
                    '1P_cam': 9,
                }
            else:
                self.camera = {
                    'facecam': 6,
                    'eyecam': 7,
                    '1P_cam': 8,
                }

        def __getitem__(self, item: CAMERA_TYPE) -> CamEvent:
            if item not in self.camera:
                raise KeyError('cam id not found')
            return CamEvent(item, self.rig._event(self.camera[item]))

    @property
    def camera_event(self) -> CameraEvent:
        """camera event. including {'facecam', 'eyecam', '1P_cam'} implemented by __getitem__()"""
        return self.CameraEvent(self)

    # ========================= #
    # Access Other output Files #
    # ========================= #

    @property
    def prot_file(self) -> Path:
        """protocol file path"""
        return self.riglog_file.with_suffix('.prot')

    @abc.abstractmethod
    def get_protocol(self) -> P:
        """
        get protocol (TypeVar ``P``)

        :return: :class:`~stimpyp.base.AbstractStimProtocol()`
        """
        pass

    def get_stimulus_type(self) -> str:
        """get stimulus type name based on protocol"""
        return self.get_protocol().stimulus_type

    @property
    def pref_file(self) -> Path:
        """preferences file path"""
        return self.riglog_file.with_suffix('.prefs')

    def get_preferences(self) -> PreferenceDict:
        """get preferences file"""
        return load_preferences(self.pref_file)

    @abc.abstractmethod
    def get_stimlog(self, *args) -> S:
        """get stimlog (TypeVar ``S``)

        :return: :class:`~AbstractStimlog`
        """
        pass


# =========== #
# Stimlog ABC #
# =========== #

class AbstractStimlog(Generic[R], metaclass=abc.ABCMeta):
    """ABC for stimulation logging. i.e., .log or .stimlog

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        P = number of acquisition sample pulse

    """

    # =========== #
    # Log Headers #
    # =========== #
    config = {}
    """config for non-header information"""

    log_info = {}
    """code:log name dictionary"""

    log_header = {}
    """code:log header dictionary"""

    # =========== #
    # Common Attr #
    # =========== #

    time: np.ndarray
    """acquisition time in sec. Array[float, P]"""

    stim_index: np.ndarray
    """stimulation index. Array[int, P]"""

    trial_index: np.ndarray
    """trial index. Array[int, P]"""

    contrast: np.ndarray
    """stimulus contrast. Array[int, P]. value domain in (0,1)"""

    frame_index: np.ndarray
    """stimulation frame index, recount every N. Array[int, P]"""

    # ======= #
    # Grating #
    # ======= #

    ori: np.ndarray
    """direction in deg. Array[float, P]"""

    sf: np.ndarray
    """spatial frequency in cyc/deg. Array[float, P]"""

    tf: np.ndarray
    """temporal frequency in hz. Array[float, P]"""

    phase: np.ndarray
    """stimulus phase for each N. Array[float, P]"""

    # ============= #
    # Function Base #
    # ============= #

    pos_x: np.ndarray
    """object center position X. Array[float, P]"""

    pos_y: np.ndarray
    """object center position Y. Array[float, P]"""

    size_x: np.ndarray
    """object size width. Array[int, P]"""

    size_y: np.ndarray
    """object size height. Array[int, P]"""

    # ====== #
    # Others #
    # ====== #

    flick: np.ndarray
    """TODO. Array[int, P]"""

    interpolate: np.ndarray
    """whether do the interpolate, Array[bool, P]"""

    mask: np.ndarray
    """TODO. Array[bool|None, P]"""

    pattern: np.ndarray
    """object pattern. Array[str, P]"""

    def __init__(self, riglog: R,
                 file_path: PathLike | None,
                 reset_mapping: dict[int, list[str]] | None = None):
        """
        :param riglog: :class:`Baselog`
        :param file_path: filepath of stimlog. could be None if shared log (pyvstim case)
        :param reset_mapping: Customized mapping for ``reset()``
            - key: corresponding to :attr:`log_header`
            - value: list of field, should be the same name as class annotations
        """
        self.riglog_data = riglog
        if file_path is not None:
            self.stimlog_file = Path(file_path)

        self._reset_mapping = reset_mapping

    @abc.abstractmethod
    def _reset(self) -> None:
        """used for assign attributes"""
        pass

    def _reset_cust_mapping(self):
        pass

    # ============ #
    # As Dataframe #
    # ============ #

    @abc.abstractmethod
    def get_visual_stim_dataframe(self, **kwargs) -> pl.DataFrame:
        """Visual presentation dataframe"""
        pass

    @abc.abstractmethod
    def get_state_machine_dataframe(self) -> pl.DataFrame:
        """State Machine dataframe"""
        pass

    def get_photo_indicator_dataframe(self) -> pl.DataFrame:
        """Photo Indicator dataframe. Github version only"""
        pass

    def get_log_dict_dataframe(self) -> pl.DataFrame:
        """Log dict dataframe. Github version only"""
        pass

    # ========= #
    # Time Info #
    # ========= #

    @property
    @abc.abstractmethod
    def exp_start_time(self) -> float:
        """experimental start time (in sec, synced to riglog time with diode offset)"""
        pass

    @property
    @abc.abstractmethod
    def exp_end_time(self) -> float:
        """experimental end time (in sec, synced to riglog time with diode offset)"""
        pass

    @property
    def stim_start_time(self) -> float:
        """the first stimulation start time (in sec, synced to riglog time with diode offset)"""
        return float(self.stimulus_segment[0, 0])

    @property
    def stim_end_time(self) -> float:
        """the last stimulation end time (in sec, synced to riglog time with diode offset)"""
        return float(self.stimulus_segment[-1, 1])

    @property
    @abc.abstractmethod
    def stimulus_segment(self) -> np.ndarray:
        """stimulation time segment (on-off) in sec (N, 2)"""
        pass

    def stim_square_pulse_event(self, sampling_rate: float = 30.) -> RigEvent:
        """
        Get the stimulation on-off square pulse 0,1 consecutive event

        :param sampling_rate: sampling rate for the time domain interpolation
        :return: Stimulus rig event
        """
        start_time = self.exp_start_time
        end_time = self.exp_end_time
        seg = self.stimulus_segment

        t = np.arange(start_time, end_time, 1 / sampling_rate)
        ret = np.zeros_like(t)
        for (on, off) in seg:
            mask = np.logical_and(on < t, t < off)
            ret[mask] = 1

        return RigEvent('visual_stim', np.vstack((t, ret)).T)

    @abc.abstractmethod
    def session_trials(self) -> dict[Session, SessionInfo]:  # TODO might not generic enough
        """get the session:SessionInfo dictionary (experimental and user-specific)"""
        pass

    @property
    @abc.abstractmethod
    def time_offset(self) -> float | np.ndarray:
        """time (in sec) to sync stimlog time to riglog"""
        pass

    # =========================== #
    # Stim Trial/Index/Cycle Info #
    # =========================== #

    @property
    def n_cycles(self) -> list[int]:
        """Number of cycle for each trial"""
        raise [1]

    @property
    @abc.abstractmethod
    def profile_dataframe(self) -> pl.DataFrame:
        """
        Dataframe with columns:

        - stim type index: ``i_stims``
        - trial index: ``i_trials``
        """
        pass

    # ================= #
    # Stim Pattern Info #
    # ================= #

    @abc.abstractmethod
    def get_stim_pattern(self, **kwargs) -> AbstractStimulusPattern:
        """get pattern foreach stimulation"""
        pass


# ================= #
# Protocol File ABC #
# ================= #

class AbstractStimProtocol(metaclass=abc.ABCMeta):
    """ABC class for the stimpy protocol file (.prot)

    `Dimension parameters`:

        N = numbers of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = Number of Stim Type

        C = Numbers of Cycle
    """

    name: str
    """protocol name. related to filename"""

    options: dict[str, Any]
    """protocol options"""

    visual_stimuli_dataframe: pl.DataFrame
    """visual stimuli dataframe. row number: S"""

    version: STIMPY_SOURCE_VERSION
    """date of major changes"""

    def __repr__(self):
        ret = list()

        ret.append('# general parameters')
        for k, v in self.options.items():
            ret.append(f'{k} = {v}')
        ret.append('# stimulus conditions')
        ret.append('\t'.join(self.stim_params))

        ret.append(printdf(self.visual_stimuli_dataframe, do_print=False))

        return '\n'.join(ret)

    def __init__(self, name: str,
                 options: dict[str, Any],
                 visual_stimuli: pl.DataFrame,
                 version: STIMPY_SOURCE_VERSION):

        self.name = name
        self.options = options
        self.visual_stimuli_dataframe = visual_stimuli
        self.version = version

        self._visual_stimuli = None

    @property
    def n_stimuli(self) -> int:
        """number of stimuli (S)"""
        return self.visual_stimuli_dataframe.shape[0]

    @property
    def stim_params(self) -> tuple[str, ...]:
        return tuple(self.visual_stimuli_dataframe.columns)

    @overload
    def __getitem__(self, item: int) -> pl.DataFrame:
        """get row of visual stimuli"""
        pass

    @overload
    def __getitem__(self, item: str) -> np.ndarray:
        """get header of stimuli"""
        pass

    def __getitem__(self, item: str | int) -> np.ndarray | pl.DataFrame:
        """Get protocol value of parameter *item*

        :param item: parameter name. either the row number(int) or field name (str)
        :return: protocol value
        :raises TypeError: `item` not a `str` or `int`
        """
        if isinstance(item, str):
            try:
                return self.visual_stimuli_dataframe.get_column(item).to_numpy()
            except ColumnNotFoundError as e:
                print(f'INVALID: {item}, select from {tuple(self.visual_stimuli_dataframe.columns)}')
                raise e

        elif isinstance(item, int):
            ret = {
                h: self.visual_stimuli_dataframe.row(item)[i]
                for i, h in enumerate(self.visual_stimuli_dataframe.columns)
            }
            return pl.DataFrame(ret)
        else:
            raise TypeError(f'{type(item)}')

    @classmethod
    @abc.abstractmethod
    def load(cls, file: Path | str) -> Self:
        r"""
        Load \*.prot file

        :param file: file path
        """
        pass

    @property
    @abc.abstractmethod
    def is_shuffle(self) -> bool:
        """if shuffle stimulation"""
        pass

    @property
    @abc.abstractmethod
    def background(self) -> float:
        """background for non-stimulation epoch.
        Note the default value need to check in user-specific stimpy version
        """
        pass

    @property
    @abc.abstractmethod
    def start_blank_duration(self) -> int:
        """blank duration before starting the visual stimulation epoch (in sec)"""
        pass

    @property
    @abc.abstractmethod
    def blank_duration(self) -> int:
        """blank duration between each visual stimulus (in sec)"""
        pass

    @property
    @abc.abstractmethod
    def trial_blank_duration(self) -> int:
        """blank duration between trials (in sec)
        TODO check stimpy source code"""
        pass

    @property
    @abc.abstractmethod
    def end_blank_duration(self) -> int:
        """blank duration after the visual stimulation epoch"""
        pass

    @property
    @abc.abstractmethod
    def trial_duration(self) -> int:
        """trial duration"""
        pass

    @property
    def visual_duration(self) -> int:
        """total visual duration"""
        return self.trial_duration * self.n_trials

    @property
    def total_duration(self) -> int:
        """total protocol duration"""
        return self.start_blank_duration + self.visual_duration + self.end_blank_duration

    @property
    def stimulus_type(self) -> str:
        """stimulus type"""
        return self.options['stimulusType']

    @property
    def n_trials(self) -> int:
        """(T,)"""
        return self.options['nTrials']


# ==================== #
# Stimulus Pattern ABC #
# ==================== #


ST = TypeVar('ST')  # Individual Stim


class AbstractStimulusPattern(Generic[ST], metaclass=abc.ABCMeta):
    """
    Abstract Stimulus Pattern

    `Dimension parameters`:

        N = numbers of visual stimulation (on-off pairs) = (T * S)
    """

    time: np.ndarray
    """stim on-off in sec. Array[float, [N, 2]]"""
    contrast: np.ndarray
    """stimulus contrast. Array[float, N]"""
    duration: np.ndarray
    """theoretical duration in prot file, not actual detected using diode. Array[float, N]"""

    def __init__(self, time: np.ndarray,
                 contrast: np.ndarray, *,
                 duration: np.ndarray | None = None):
        """

        :param time: stim on-off in sec. Array[float, [N, 2]]
        :param contrast: stimulus contrast. Array[float, N]
        :param duration: theoretical duration in prot file, not actual detected using diode. Array[float, N]
        """
        self.time = time
        self.contrast = contrast
        self.duration = duration

    @classmethod
    def of(cls, rig: R) -> Self:
        """
        init from Baselog children class

        :param rig: :class:`~stimpyp.base.AbstractLog`
        :return: :class:`StimPattern`
        """
        return rig.get_stimlog().get_stim_pattern()

    @abc.abstractmethod
    def foreach_stimulus(self, name: bool = False) -> Iterable[tuple[Any, ...] | ST]:
        pass
