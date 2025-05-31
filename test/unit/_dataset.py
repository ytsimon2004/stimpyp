import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, ContextManager

import gdown

from stimpyp import STIMPY_SOURCE_VERSION, RiglogData, PyVlog, load_camlog, CAMERA_VERSION

__all__ = ['load_example_riglog',
           'load_example_camlog']


def load_example_riglog(source_version: STIMPY_SOURCE_VERSION, *,
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


def load_example_camlog(source_version: CAMERA_VERSION, cached: bool = True):
    match source_version:
        case 'labcams':
            file_id = '1u8mqFZxv6AbKiKtswJBbzPLyBAUYSE2_'
        case 'pycams':
            file_id = '1coAAjycdQEXzMw36CVdOYPtcELlkDnRQ'
        case _:
            raise ValueError('')

    with google_drive_file(file_id, cached=cached) as f:
        return load_camlog(f, camera_version=source_version)


def joinn(sep: str, *part: str | None) -> str:
    """join non-None str with sep."""
    return sep.join([str(it) for it in part if it is not None])


@contextmanager
def google_drive_folder(folder_id: str, *,
                        quiet: bool = False,
                        rename_folder: str | None = None,
                        cached: bool = False,
                        invalid_cache: bool = False) -> ContextManager[Path]:
    if rename_folder is not None:
        folder_name = rename_folder
    else:
        folder_name = folder_id

    output_dir = Path('test_data') / folder_name

    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not invalid_cache:
            yield output_dir
        else:
            output_dir.mkdir(exist_ok=True, parents=True)
            gdown.download_folder(id=folder_id, output=str(output_dir), quiet=quiet)
            yield output_dir
    finally:
        if not cached:
            shutil.rmtree(output_dir, ignore_errors=True)


@contextmanager
def google_drive_file(file_id: str,
                      *,
                      quiet: bool = False,
                      rename_file: str | None = None,
                      cached: bool = False,
                      invalid_cache: bool = False) -> ContextManager[Path]:
    """
    Download file from Google Drive. If not ``cached``, then delete afterward.

    :param file_id: Google Drive file ID used to identify the file to be downloaded.
    :param quiet: Boolean flag to suppress output from the download process.
    :param rename_file: Optional string to rename the downloaded file.
    :param cached: Boolean flag to retain the downloaded file after usage.
    :param invalid_cache: Boolean flag to force re-download even if the file exists in the cache.
    :return: A context manager yielding the path to the downloaded file.
    """
    if rename_file is not None:
        file = rename_file
    else:
        file = file_id

    output_file = Path('test_data') / file

    try:
        if output_file.exists() and not invalid_cache:
            yield output_file
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_file), quiet=quiet)
            yield output_file
    finally:
        if not cached:
            output_file.unlink(missing_ok=True)
