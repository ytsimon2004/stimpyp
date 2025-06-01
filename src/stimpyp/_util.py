import functools
import warnings
from pathlib import Path

import pandas as pd
import polars as pl

__all__ = ['try_casting_number',
           'unfold_stimuli_condition',
           'deprecated_func',
           'printdf',
           'uglob',
           'cls_hasattr']


def try_casting_number(value: str, do_eval: bool = False) -> float | int | str:
    """
    try casting of string to numerical value

    :param value: str
    :param do_eval: do built in eval()
    :return:
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if do_eval:
                try:
                    return eval(value)
                except (SyntaxError, NameError):
                    pass

    return value


def unfold_stimuli_condition(parts: list[str]) -> list[list]:
    """
    unfold the numbers of stimuli (each row in prot file) for parsing

    :param parts: list of str foreach row in the visual stimuli dataframe

        e.g., ['1', '3', '0', '0', '1', '0.04', '0', '0', '200', '200', "{'phase':['linear',1]}"]
    """
    ret = []
    if '-' in parts[0]:
        nstim = list(map(int, parts[0].split('-', maxsplit=1)))
        nstim = tuple((nstim[0], nstim[1] + 1))
        ranging = nstim[1] - nstim[0]  # how many n in each row

        for n in range(*nstim):
            parts[0] = f'{n}'
            ext = []
            for i, it in enumerate(parts):
                if '{i}' in it:
                    it = it.replace('{i}', f'{n - 1}%{ranging}')
                if '{t}' in it:
                    raise NotImplementedError('')

                ext.append(try_casting_number(it, do_eval=True))

            ret.append(ext)

    else:
        for i, it in enumerate(parts):
            ret.append(try_casting_number(it, do_eval=True))

        ret = [ret]
    return ret


def deprecated_func(*, new: str | None = None,
                    remarks: str | None = None,
                    removal_version: str = None):
    """Mark deprecated functions.

    :param new: The renamed new usage
    :param remarks: The reason why the function is deprecated
    :param removal_version: Optional version or date when the function is planned to be removed
    """

    def _decorator(f):

        @functools.wraps(f)
        def _deprecated_func(*args, **kwargs):
            msg = _build_deprecated_message(f.__qualname__, new, remarks, removal_version)

            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2
            )

            return f(*args, **kwargs)

        if f.__doc__ is None:
            _deprecated_func.__doc__ = f"DEPRECATED ({remarks}). "
        else:
            _deprecated_func.__doc__ = f"DEPRECATED ({remarks}). " + f.__doc__

        return _deprecated_func

    return _decorator


def _build_deprecated_message(target: str,
                              alternation: str | None = None,
                              remarks: str | None = None,
                              removal: str | None = None) -> str:
    msg = f'{target} is deprecated'

    if removal is not None:
        msg += f' and will be removed in a future release (after version {removal}).'
    else:
        msg += '.'

    if alternation is not None:
        msg += f' Please use "{alternation}" instead.'

    if remarks is not None:
        msg += f' NOTE: {remarks}.'

    return msg


def printdf(df: pl.DataFrame | pd.DataFrame,
            nrows: int | None = None,
            ncols: int | None = None,
            do_print: bool = True,
            **kwargs) -> str:
    """
    print dataframe with given row numbers (polars)
    if isinstance pandas dataframe, print all.

    :param df: polars or pandas dataframe
    :param nrows: number of rows (applicable in polars case)
    :param ncols: number of columns
    :param do_print: do print otherwise, only return the str
    :param kwargs: additional arguments pass the ``pl.Config()``
    :return:
    """

    if isinstance(df, pl.DataFrame):
        with pl.Config(**kwargs) as cfg:
            rows = df.shape[0] if nrows is None else nrows
            cols = df.shape[1] if ncols is None else ncols
            cfg.set_tbl_rows(rows)
            cfg.set_tbl_cols(cols)

            if do_print:
                print(df)

            return df.__repr__()

    elif isinstance(df, pd.DataFrame):
        ret = df.to_markdown()
        print(ret)
        return ret

    else:
        raise TypeError('')


def uglob(directory: Path,
          pattern: str,
          is_dir: bool = False) -> Path:
    """
    Use glob pattern to find the unique file in the directory.

    :param directory: Directory
    :param pattern: Glob pattern
    :param is_dir: Is the pattern point to a directory?
    :return: The unique path
    :raise FileNotFoundError: the unique path is not existed.
    :raise NotADirectoryError: *directory* is not a directory
    :raise RuntimeError: more than one path are found.
    """

    if not directory.exists():
        raise FileNotFoundError(f'{directory} not exit')

    if not directory.is_dir():
        raise NotADirectoryError(f'{directory} is not a directory')

    f = list(directory.glob(pattern))

    if is_dir:
        f = [ff for ff in f if ff.is_dir()]
    else:
        f = [ff for ff in f if not ff.is_dir()]

    if len(f) == 0:
        t = 'directory' if is_dir else 'file'
        raise FileNotFoundError(f'{directory} not have {t} with the pattern: {pattern}')
    elif len(f) == 1:
        return f[0]
    else:
        f.sort()
        t = 'directories' if is_dir else 'files'
        raise RuntimeError(f'multiple {t} were found in {directory} with the pattern {pattern} >>> {f}')


def cls_hasattr(cls: type, attr: str) -> bool:
    """
    Check if attributes in class

    :param cls: The class to check for the attribute.
    :param attr: The name of the attribute to look for within the class and its hierarchy.
    :return: True if the class or any of its parent classes has the specified attribute, False otherwise.
    """
    if attr in getattr(cls, '__annotations__', {}):
        return True

    for c in cls.mro()[1:]:  # Skip the first class as it's already checked
        if attr in getattr(c, '__annotations__', {}):
            return True

    return False
