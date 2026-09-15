"""Checkpoint A's signed-log display, with physical residual tick labels."""
import numpy as np

FLOOR = 1e-16
E_MIN = -16.0


def signed_log(residual):
    """Match expA06_readout_structure/app.py::_symlog exactly.

    Zero and magnitudes <= 1e-16 map to the center. Above that floor,
    every decade occupies one unit, with the original residual's sign.
    """
    residual = np.asarray(residual, dtype=np.float64)
    magnitude = np.abs(residual)
    exponent = np.where(magnitude > 0,
                        np.log10(np.where(magnitude > 0, magnitude, 1.0)), E_MIN)
    return np.sign(residual) * np.clip(exponent - E_MIN, 0.0, None)


def set_signed_log_axis(ax, transformed_max):
    """Use the app's symmetric limits, with fewer labels for compact panels."""
    extent = max(2.0, float(transformed_max) * 1.08)
    stride = 1 if extent <= 8 else 4
    ticks, labels = [0.0], ["0"]
    for exponent in range(int(E_MIN) + stride, int(np.ceil(E_MIN + extent)) + 1, stride):
        position = exponent - E_MIN
        if position <= extent:
            ticks.extend([position, -position])
            labels.extend([rf"$10^{{{exponent}}}$", rf"$-10^{{{exponent}}}$"])
    order = np.argsort(ticks)
    ax.set_yticks(np.asarray(ticks)[order], labels=np.asarray(labels)[order])
    ax.set_ylim(-extent, extent)

