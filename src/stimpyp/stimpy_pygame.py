import collections
import logging
from pathlib import Path
from typing import Literal, TypedDict, NamedTuple, Any

import numpy as np
import polars as pl
from typing_extensions import Self

from stimpyp._type import PathLike
from stimpyp.event import RigEvent
from stimpyp.session import Session, SessionInfo, get_protocol_sessions
from stimpyp.stimpy_core import RiglogData
from stimpyp.stimpy_git import load_stimlog

__all__ = ['PyGameLinearStimlog',
           'WorldMapInfo']

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

        if riglog is not None:
            self.stimlog_data = load_stimlog(riglog.stimlog_file)
        else:
            self.stimlog_data = load_stimlog(filepath)

        # diode offset
        self.diode_offset = diode_offset
        self.offset_method = offset_method
        self._offset_time = None

        # cache
        self._total_length = None

    @property
    def exp_start_time(self) -> float:
        return float(self.riglog_data.dat[0, 2] / 1000)

    @property
    def passive_start_time(self) -> float:
        df = self.get_state_machine_dataframe()
        return df.filter(pl.col('state') == 'RUN_PASSIVE')['time'].min()

    @property
    def exp_end_time(self) -> float:
        return float(self.riglog_data.dat[-1, 2] / 1000)

    @property
    def virtual_position_event(self) -> RigEvent:
        df = self.get_agent_dataframe()

        # replace x to min position in the track
        start_x = df.filter(pl.col('touch') == 'start')['x'].min()
        df = df.with_columns(
            pl.when(pl.col('x') <= start_x)
            .then(start_x)
            .otherwise(pl.col('x'))
            .alias('x')
        )

        # map (0 to max)
        df = df.with_columns(pl.col('x') - start_x)

        return RigEvent('position', np.vstack((df['time'], (df['x']))).T)

    @property
    def virtual_lap_event(self) -> RigEvent:
        df = self.get_log_dict_dataframe()
        return RigEvent('lap', np.vstack((df['time'], (df['trial_nr']))).T)

    @property
    def screen_event(self) -> RigEvent:
        df = self.get_photo_indicator_dataframe()
        return RigEvent('screen', np.vstack((df['time'], (df['state']))).T)

    @property
    def offset_time(self) -> float | np.ndarray:
        if self._offset_time is None:
            raise ValueError('offset time is not calculated by _offset() yet')
        return self._offset_time

    def session_trials(self) -> dict[Session, SessionInfo]:
        return {
            prot.name: prot
            for prot in get_protocol_sessions(self)
        }

    def get_agent_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['Agent'])

    def get_photo_indicator_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['PhotoIndicator'])

    def get_log_dict_dataframe(self) -> pl.DataFrame:
        df = self.stimlog_data['LogDict']
        df = (
            df.with_columns(pl.col('trial_nr').str.replace('None', '-1').cast(pl.Int64))
            .filter(pl.col('trial_nr') >= 0)  # remove initial
            .with_columns(pl.col('trial_nr') + 1)  # match riglog value starting from 1
        )

        return self._offset(df)

    def get_state_machine_dataframe(self) -> pl.DataFrame:
        return self._offset(self.stimlog_data['DispatchStateMachine'])

    def _offset(self, df: pl.DataFrame) -> pl.DataFrame:

        if self.riglog_data is None:
            logger.warning('can not do offset if lack of riglog')
            return df

        if self.diode_offset:
            match self.offset_method:
                case 'scalar':
                    if self._offset_time is None:
                        self._offset_time = _diode_offset_scalar(self.riglog_data, self.stimlog_data)
                    dt = self._offset_time
                    return df.with_columns((pl.col('time') - dt).alias('time'))
                case 'vector':
                    return self._trial_base_offset(df)[1]
                case _:
                    raise ValueError('Unknown offset method')
        else:
            self._offset_time = 0

        return df

    def _trial_base_offset(self, df: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        event_time = df['time'].to_numpy()
        trial_time = self.stimlog_data['LogDict']['time'][1:].to_numpy().copy()
        dt = _diode_offset_vector(self.riglog_data, self.stimlog_data)
        self._offset_time = dt

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

        return dt, df.with_columns(pl.Series('time', corrected_time))

    def get_max_virtual_length(self) -> np.ndarray:
        """`Array[float, L]`"""
        return self.get_agent_dataframe().filter(pl.col('touch') == 'reward')['x'].to_numpy()

    def get_virtual_length(self, count: int = 1, length: float = 150) -> float:
        """get actual virtual trial length based on riglog encoder value mapping

        :param count: number of photosensing for each trial for rig encoder
        :param length: length in cm for each trial for rig encoder
        """
        if self._total_length is None:
            self._total_length = self._get_virtual_length(count, length)
        return self._total_length

    def _get_virtual_length(self, count: int = 1, length: float = 150) -> float:
        """get actual virtual trial length based on riglog encoder value mapping"""
        if not self.diode_offset:
            raise ValueError('diode offset need to be set')

        rig = self.riglog_data
        rig_pos = rig.position_event
        unwarp_enc = rig.unwarp_circular_position()

        # use second lap time
        v_start = self.get_log_dict_dataframe().filter(pl.col('trial_nr') == 2)['time'].item()
        v_end = self.get_agent_dataframe().filter(pl.col('touch') == 'reward')['time'][1]

        mx = np.logical_and(rig_pos.time >= v_start, rig_pos.time < v_end)
        ret = unwarp_enc[mx]

        f, _ = self.riglog_data.get_encoder_factor(count, length)

        return (ret.max() - ret.min()) * f

    def get_landmarks(self, row: int = 0, char: str = 'v') -> list[np.ndarray]:
        """
        Get landmark locations grouped by consecutive sequences.
        
        Returns a list of arrays, where each array contains the positions 
        of a consecutive sequence of the specified character.
        """
        total = self.get_virtual_length()
        world = self.riglog_data.get_worldmap()
        index_groups = linear_texture_index(world, row, char)

        if len(index_groups) == 0:
            return []

        # to position coordinates
        landmark_groups = []
        for group in index_groups:
            positions = np.array([i / len(world.world_map[0]) for i in group]) * total
            landmark_groups.append(positions)

        return landmark_groups


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


class TextureInfo(TypedDict, total=False):
    texture: str
    event: str
    desp: str


class WorldMapInfo(NamedTuple):
    """
    World map configuration for pygame-based virtual reality environments.
    
    This class represents a parsed world map with texture mapping, position markers,
    and spatial configuration parameters. Maps are defined in .map files with the format:
    
    ```
    # C : texture=name, event=type, desp='description'
    # parameter = value
    MAP_GRID_DATA
    ```
    
    Texture Mapping:
    - Numeric codes (0-3): dot textures (dot0, dot1, dot2, dot3)
    - Letter codes: special textures (g=gray, v=vertical, d=diagonal)
    - Position markers: P=reward, Q=start, R=teleport
    
    Example configuration:
    ```
    # texture_path = .
    # 0 : texture=dot0
    # 1 : texture=dot1
    # g : texture=gray
    # P : desp='reward position', event=reward
    # grid_size = 100
    # wall_height = 200
    ```
    """

    texture_path: Path
    """Base path for texture files"""

    world_map: np.ndarray
    """Array[H, W] - 2D grid where each cell contains texture/marker codes"""

    info_pos: dict[int, list[tuple[int, int]]]
    """Mapping from character codes to grid positions: {ord(char) -> [(x, y)]}"""

    texture_info: dict[int, TextureInfo]
    """Texture configuration for each character code: {ord(char) -> TextureInfo}"""

    parameters: dict[str, str]
    """Map configuration parameters (grid_size, wall_height, texture_path, etc.)"""

    @property
    def width(self) -> int:
        """W"""
        return self.world_map.shape[1]

    @property
    def height(self) -> int:
        """H"""
        return self.world_map.shape[0]

    @property
    def grid_size(self) -> int:
        return int(self.parameters.get('grid_size', '100'))

    @property
    def wall_height(self) -> int:
        return int(self.parameters.get('wall_height', '100'))

    @property
    def info_pos_char(self) -> dict[str, list[tuple[int, int]]]:
        """{C -> [(x, y)]}"""

        class GetItem:
            def __len__(zelf):
                return len(self.info_pos)

            def __contains__(zelf, item):
                return ord(item) in self.info_pos

            def __getitem__(zelf, item):
                return self.info_pos[ord(item)]

        return GetItem()

    @classmethod
    def load_map(cls, path: PathLike) -> Self:
        return load_map(path)

    def align_grid(self, x: float, y: float) -> tuple[int, int]:
        return align_grid(x, y, self.grid_size)


def align_grid(x: float, y: float, g: float) -> tuple[int, int]:
    return int(x / g), int(y / g)


def load_map(path: PathLike) -> WorldMapInfo:
    """
    Format of worldmap
    ==================

    ```
    # C : key=value, ...
    # P = value
    MAP
    ```

    where `C` is a single character that present as a texture block.
    The texture parameters are parsed as dict's interior expression.
    If `C` is empty, then it means background.
    `P` is a parameter name
    `MAP` is a world map built by `C`s.

    :param path:
    :return:
    """
    if isinstance(path, str):
        if '/' in path:
            path = Path(path)
        else:
            path = Path(__file__).parent / 'world_map' / path

    if path.suffix != '.map':
        raise RuntimeError()

    world_map: list[list[int]] = []
    texture_info: dict[int, dict[str, str]] = {}
    parameters: dict[str, str] = {}

    with path.open() as f:
        for line, content in enumerate(f):
            content = content.strip()

            try:
                _load_map_line(content, world_map, texture_info, parameters)
            except RuntimeError as e:
                raise RuntimeError(f'line {line}: "{content}"') from e

    # remove tailing empty lines
    if len(world_map) > 0 and len(world_map[-1]) == 0:
        del world_map[-1]

    world_width = max([len(it) for it in world_map])
    for world_row in world_map:
        if (dd := world_width - len(world_row)) > 0:
            world_row.extend([0] * dd)
    world_map: np.ndarray = np.array(world_map)

    #
    info_pos: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for a, b in texture_info.items():
        y, x = np.nonzero(world_map == a)
        for (yy, xx) in zip(y, x):
            info_pos[a].append((xx, yy))
            world_map[yy, xx] = a

    world_map[world_map == ord(' ')] = 0  # space to 0

    #
    texture_path = path.parent

    try:
        _texture_path = Path(parameters['texture_path'])
    except (KeyError, ValueError):
        pass
    else:
        if _texture_path.is_absolute():
            texture_path = _texture_path
        else:
            texture_path = texture_path / _texture_path

    return WorldMapInfo(texture_path, world_map, dict(info_pos), texture_info, parameters)


class MissingSymbolAsStr(collections.defaultdict):
    def __missing__(self, key):
        return key


def _load_map_line(content: str, world_map: list[list[int]],
                   texture_info: dict[int, dict[str, Any]],
                   parameters: dict[str, str]):
    if content.startswith('#') and ':' in content:
        expr, _, desp = content[1:].partition(':')
        expr = expr.strip()
        desp = desp.strip()

        if len(expr) == 0:
            code = ord(' ')
        elif len(expr) == 1:
            code = ord(expr)
        else:
            raise RuntimeError(f'texture code over 1-len {expr}')

        desp = eval('dict(' + desp + ')', {}, MissingSymbolAsStr(dict=dict))
        texture_info[code] = desp

    elif content.startswith('#') and '=' in content:
        expr, _, desp = content[1:].partition('=')
        expr = expr.strip()
        desp = desp.strip()
        parameters[expr] = desp

    elif content.startswith('#'):
        pass

    elif len(content) > 0:
        world_map.append(list(map(ord, content)))
    else:
        world_map.append([])


def linear_texture_index(world: WorldMapInfo, row: int, char: str) -> list[list[int]]:
    """
    Get indices of character in a row, grouped by consecutive sequences.
    
    Example:
        If 'v' appears at positions [5, 6, 7, 15, 16, 20], returns [[5, 6, 7], [15, 16], [20]]
    """
    row_str = [chr(code) for code in world.world_map[row]]
    indices = [i for i, s in enumerate(row_str) if s == char]

    if not indices:
        logger.error(f'no char: {char} in row: {row} in the world map')
        return []

    # Group consecutive indices
    groups = []
    current_group = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:  # consecutive
            current_group.append(indices[i])
        else:  # not consecutive, start new group
            groups.append(current_group)
            current_group = [indices[i]]

    # Add the last group
    groups.append(current_group)

    return groups
