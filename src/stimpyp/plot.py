from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

__all__ = [
    'plot_signal_diff',
    'plot_scatter_animation'
]


def plot_signal_diff(t: np.ndarray,
                     alpha: float = 0.6,
                     label: str | None = None,
                     ax: Axes | None = None,
                     **kwargs):
    """
    Plots the reciprocal of the difference of subsequent elements in the input
    array ``t``. This is commonly used for visualizing a derived signal or property
    based on the change between consecutive data points in the input array.
    Additional configurations can be applied to the plot via keyword arguments
    (**kwargs).

    :param t: Input array representing the signal or data points.
        The function computes the reciprocal of the difference between consecutive
        elements.
    :param alpha: Transparency of the plot.
    :param label: Label for the plot.
    :param ax: Matplotlib Axes object to be used for plotting. If None is provided,
        a new Axes instance will be created.
    :param kwargs: Additional keyword arguments for customizing the plot. These
        are passed to the set() method of the Axes object.
    :return: The Axes object containing the plot. If a new Axes instance was
        created internally, it is returned here.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(t, 1 / np.diff(t, append=t[0]), alpha=alpha, label=label)
    ax.set(**kwargs)


def plot_scatter_animation(x: np.ndarray,
                           y: np.ndarray,
                           t: np.ndarray | None = None, *,
                           step: int | None = None,
                           size: int = 10,
                           output: Path | str | None = None,
                           **kwargs) -> None:
    r"""
    Plot xy scatter animation with given time points

    :param x: x loc. `Array[float, T]`
    :param y: y loc. `Array[float, T]`
    :param t: time array in sec. `Array[float, T]`
    :param size: size of the scatter
    :param step: step run per datapoint
    :param output: output for animation. i.e., \*.gif
    :param kwargs: additional arguments passed to ``FuncAnimation()``
    :return:
    """
    from matplotlib.animation import FuncAnimation

    fig, _ = plt.subplots()

    def foreach_run(frame: int):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))

        if step is not None:
            frame *= step

        ax.text(0.02, 0.95, f'Frames = {frame}', transform=ax.transAxes)
        if t is not None:
            ax.text(0.02, 0.85, f'Time = {t[frame]:.2f}', transform=ax.transAxes)

        ax.scatter(x[frame], y[frame], s=size)

    ani = FuncAnimation(fig, foreach_run, frames=len(x), **kwargs)

    try:
        if output is not None:
            ani.save(output)
        else:
            plt.show()

    finally:
        plt.clf()
        plt.close('all')
