from typing import Literal

from neuralib.io.dataset import google_drive_folder
from neuralib.util.utils import joinn
from stimpyp.parser import STIMPY_SOURCE_VERSION, PyVlog, RiglogData

__all__ = ['load_example_data']


def load_example_data(source_version: STIMPY_SOURCE_VERSION, *,
                      stim_type: Literal['sftfdir', 'circular'] | None = None,
                      cached: bool = True) -> RiglogData | PyVlog:
    """
    Load log data for treadmill task with optional visual stimulation

    :param source_version: ``STIMPY_SOURCE_VERSION``
    :param stim_type:
    :param cached: Set as True if used other files (.prot, .stimlog) after create the instance
    :return:
    """
    name = 'riglog'

    match source_version, stim_type:
        case ('stimpy-bit', None):
            folder = "1VaZ1x7BiMwt2s5ZbkgYB6hmKEue3w3_d"
            cls = RiglogData
        case ('stimpy-bit', 'sftfdir'):
            folder = "1Kjr-tgZd-11Lm5ZAswmOxXJ1h-hXszBr"
            cls = RiglogData
        case ('stimpy-git', 'sftfdir'):
            folder = "1jHSGrFLpLaEfglkFniCq2WxjOJogkD3p"
            cls = RiglogData
        case ('pyvstim', 'circular'):
            folder = '1z8ajR_7Yx4enETEVPBiF_Y2u5prk2IbI'
            cls = PyVlog
        case _:
            raise NotImplementedError('')

    name += '_' + joinn('_', source_version, stim_type)

    with google_drive_folder(folder, cached=cached, rename_folder=name) as src:
        return cls(root_path=src)
