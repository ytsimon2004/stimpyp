from typing import NamedTuple, Iterable

import numpy as np
from typing_extensions import TypeAlias, Self

from .base import AbstractStimulusPattern

__all__ = [
    'Direction',
    'SF',
    'TF',
    'SFTF',
    'VisualParas',
    #
    'GratingStim',
    'FunctionStim',
    #
    'GratingPattern',
    'FunctionPattern'
]

Direction: TypeAlias = int
"""stimulus direction in deg"""

SF: TypeAlias = float
"""spatial frequency in cyc/deg"""

TF: TypeAlias = int
"""temporal frequency in hz"""

SFTF: TypeAlias = tuple[SF, TF]
"""tuple of spatial and temporal frequencies"""

VisualParas: TypeAlias = tuple[SF, TF, Direction]
"""tuple of spatial, temporal frequencies and degree"""


# =============== #
# Individual Stim #
# =============== #


class GratingStim(NamedTuple):
    index: int
    """stimulus index"""
    time: np.ndarray
    """stim on-off time. Array[float, 2]."""
    sf: SF
    """``SF``"""
    tf: TF
    """``TF``"""
    direction: Direction
    """``Direction``"""


class FunctionStim(NamedTuple):
    index: int
    """stimulus index"""
    time: np.ndarray
    """stim on-off time. Array[float, 2]."""
    pos_xy: np.ndarray
    """object center position XY. Array[float, 2]"""
    size_xy: np.ndarray
    """object size width and height. Array[float, 2]"""


class GratingPattern(AbstractStimulusPattern):
    """Grating Stimulus Pattern"""

    direction: np.ndarray
    """stimulus direction in deg. Array[int, N]"""

    sf: np.ndarray
    """spatial frequency in cyc/deg. Array[float, N]"""

    tf: np.ndarray
    """temporal frequency in hz Array[int, N]"""

    def __init__(self,
                 time: np.ndarray,
                 contrast: np.ndarray,
                 direction: np.ndarray,
                 sf: np.ndarray,
                 tf: np.ndarray,
                 *,
                 duration: np.ndarray | None = None):
        """

        :param time: stim on-off in sec. Array[float, [N, 2]]
        :param contrast: stimulus contrast. Array[float, N]
        :param direction: stimulus direction in deg. Array[int, N]
        :param sf: spatial frequency in cyc/deg. Array[float, N]
        :param tf: temporal frequency in hz Array[int, N]
        :param duration: theoretical duration in prot file, not actual detected using diode. Array[float, N]
        """

        super().__init__(time, contrast, duration=duration)

        self.direction = direction
        self.sf = sf
        self.tf = tf

    @classmethod
    def of(cls, rig: 'R') -> Self:
        return super().of(rig)

    @property
    def sf_set(self) -> np.ndarray:
        """unique sf_set"""
        return np.array(sorted(self.sf_i().keys()))

    @property
    def tf_set(self) -> np.ndarray:
        """unique tf_set"""
        return np.array(sorted(self.tf_i().keys()))

    @property
    def n_sf(self) -> int:
        """number of sf set"""
        return len(self.sf_set)

    @property
    def n_tf(self) -> int:
        """number of tf set"""
        return len(self.tf_set)

    @property
    def n_sftf(self) -> int:
        """number of sftf combination"""
        return len(self.sftf_i())

    @property
    def n_dir(self) -> int:
        """number of direction"""
        return len(self.dir_i())

    def dir_i(self) -> dict[Direction, int]:
        """deg:index dict"""
        return {it.item(): i for i, it in enumerate(sorted(np.unique(self.direction)))}

    def sf_i(self) -> dict[SF, int]:
        """sf:index dict"""
        return {it.item(): i for i, it in enumerate(sorted(np.unique(self.sf)))}

    def tf_i(self) -> dict[TF, int]:
        """sf:index dict"""
        return {it.item(): i for i, it in enumerate(sorted(np.unique(self.tf)))}

    # previous plot use tfsf as condition idx
    def sftf_i(self) -> dict[SFTF, int]:
        """sf, tf combination. (sf , tf):y"""
        return {
            it: i
            for i, it in enumerate([
                (sf.item(), tf.item())
                for sf in sorted(np.unique(self.sf))
                for tf in sorted(np.unique(self.tf))
            ])
        }

    def sftfdir_i(self) -> dict[VisualParas, int]:
        """sf, tf, ori combination. (sf , tf , ori):y"""
        return {
            it: i  # (sf , tf, ori): index
            for i, it in enumerate([
                (sf.item(), tf.item(), ori * 30)
                for sf in sorted(np.unique(self.sf))
                for tf in sorted(np.unique(self.tf))
                for ori in range(12)
            ])
        }

    def foreach_stimulus(self, name: bool = False) -> Iterable[tuple[int, np.ndarray, SF, TF, Direction] | GratingStim]:
        """
        Generator for (index, stimulus_time, sf, tf, ori)

        :param name: If True, return ``GratingStim``, otherwise, return tuple
        :return:
        """
        for si, st in enumerate(self.time):
            ret = si, st, self.sf[si], self.tf[si], self.direction[si]

            if name:
                yield GratingStim(*ret)
            else:
                yield ret

    def get_stim_time(self) -> float:
        """get approximate stim time if the same duration. i.e., for plotting purpose"""
        return np.mean(self.time[:, 1] - self.time[:, 0])


class FunctionPattern(AbstractStimulusPattern):
    """"Function Stimulus Pattern"""

    pos_xy: np.ndarray
    """object center position XY. `Array[float, [N, 2]]`"""
    size_xy: np.ndarray
    """object size width and height. `Array[float, [N, 2]]`"""

    def __init__(self, time: np.ndarray,
                 contrast: np.ndarray,
                 pos_xy: np.ndarray,
                 size_xy: np.ndarray, *,
                 duration: np.ndarray | None = None):
        """

        :param time: stim on-off in sec. Array[float, [N, 2]]
        :param contrast: stimulus contrast. Array[float, N]
        :param pos_xy: object center position XY. Array[float, [N, 2]]
        :param size_xy: object size width and height. Array[float, [N, 2]]
        :param duration: theoretical duration in prot file, not actual detected using diode. Array[float, N]
        """
        super().__init__(time, contrast, duration=duration)

        self.pos_xy = pos_xy
        self.size_xy = size_xy

    def foreach_stimulus(
            self,
            name: bool = False
    ) -> Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray] | FunctionStim]:
        """
        Generator for (index, stimulus_time, pos_xy, size_xy)

        :param name: If True, return ``FunctionStim``, otherwise, return tuple
        :return:
        """

        for si, st in enumerate(self.time):
            ret = si, st, self.pos_xy[si], self.size_xy[si]

            if name:
                yield FunctionStim(*ret)
            else:
                yield ret
