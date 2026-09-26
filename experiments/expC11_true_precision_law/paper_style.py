"""Figure style adapted from PR #5's optimization handoff.

Source: https://github.com/samlayton99/precision-mlps/pull/5
Commit: d89561cead41a273caea95ba3b3200ffff8cf051
The three-panel construction figure uses a wider canvas for its eight-function legend.
"""
from matplotlib.lines import Line2D
import numpy as np

STYLE_SOURCE = {
    "url": "https://github.com/samlayton99/precision-mlps/pull/5",
    "commit": "d89561cead41a273caea95ba3b3200ffff8cf051",
    "file": "figure_handoff/optimization/render.py",
}
DPI = 600


def apply(fig, axes):
    """Apply the handoff's typography, strokes, grids, and frameless local legends."""
    fig.set_size_inches(9., 3.0)
    fig.subplots_adjust(left=.072, right=.99, bottom=.17, top=.89, wspace=.23)
    for index, ax in enumerate(axes):
        ax.set_title(ax.get_title(), fontsize=8.5, fontweight="bold", y=1., pad=7)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 3
        ax.tick_params(which="major", labelsize=7, length=3, width=.85, pad=3)
        ax.tick_params(which="minor", length=1.3, width=.35)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1.)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.grid(False, axis="x", which="both")
        for tick, value in zip(ax.yaxis.get_major_ticks(), ax.yaxis.get_majorticklocs()):
            tick.gridline.set_color("#D8D8D8")
            tick.gridline.set_linewidth(.45)
            tick.gridline.set_alpha(1.)
            tick.gridline.set_visible(int(round(np.log10(value))) % 2 == 0)
        for line in ax.lines:
            line.set_linewidth(1.45 if line.get_linestyle() != "--" else 1.25)
            if line.get_marker() not in ("None", "", None):
                line.set_markersize(3.1)
                line.set_markeredgewidth(.85)
                # Show distinct observations without filling the roundoff plateau with markers.
                count = len(line.get_xdata())
                line.set_markevery(sorted(set(range(0, count, 4)) | {count-1}))
        if index == 1:
            for collection in ax.collections:
                collection.set_sizes([13])
                collection.set_linewidths([.85])
        old = ax.get_legend()
        handles = old.legend_handles
        labels = [text.get_text() for text in old.get_texts()]
        title = old.get_title().get_text()
        for handle in handles:
            if isinstance(handle, Line2D):
                handle.set_linewidth(1.45 if handle.get_linestyle() != "--" else 1.25)
                handle.set_markersize(3.1)
                handle.set_markeredgewidth(.85)
        legend_size = 6.3 if index == 1 else 7
        ax.legend(handles, labels, title=title or None, title_fontsize=legend_size, fontsize=legend_size,
                  loc="lower right" if index == 1 else "upper right", ncol=2 if index == 1 else 1,
                  frameon=False, handlelength=1.5, handletextpad=.45, labelspacing=.25,
                  columnspacing=.7, borderaxespad=.4, borderpad=.2)
