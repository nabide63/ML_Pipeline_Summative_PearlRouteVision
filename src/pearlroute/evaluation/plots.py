"""Confusion matrix, learning curves and EDA charts.

Colours follow one validated palette: train = blue, validation = orange (categorical slots 1 and 2);
the confusion matrix uses a single blue ramp, light = few, dark = many.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3dd"
TRAIN = "#2a78d6"
VAL = "#eb6834"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_history(history: pd.DataFrame, title: str = "") -> Figure:
    """Loss and accuracy per epoch, train vs validation, with the best (lowest val loss) epoch marked."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    best = history["val_loss"].idxmin()
    for ax, metric, label in [(axes[0], "loss", "Loss (lower is better)"),
                              (axes[1], "accuracy", "Accuracy (higher is better)")]:
        _style(ax)
        ax.plot(history.index, history[metric], color=TRAIN, linewidth=2, label="train")
        ax.plot(history.index, history[f"val_{metric}"], color=VAL, linewidth=2, linestyle="--",
                label="validation")
        ax.axvline(best, color=TEXT_MUTED, linewidth=1, linestyle=":")
        ax.annotate(f"best epoch {best}", (best, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", color=TEXT_MUTED, fontsize=8)
        for series, colour in [(metric, TRAIN), (f"val_{metric}", VAL)]:   # direct end labels
            ax.annotate(f"{history[series].iloc[-1]:.2f}", (history.index[-1], history[series].iloc[-1]),
                        xytext=(5, 0), textcoords="offset points", va="center", color=TEXT, fontsize=8)
        ax.set_title(label, loc="left", color=TEXT)
        ax.set_xlabel("epoch", color=TEXT)
    axes[1].legend(frameon=False, loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray, class_names: list[str], title: str = "") -> Figure:
    """Rows = true class, columns = predicted class. Colour = share of the row; text = count and %."""
    cm = np.asarray(cm)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [c.replace("_", "\n") for c in class_names]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor=SURFACE)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            colour = "white" if share[i, j] > 0.55 else TEXT
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", color=colour, fontsize=8)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted class", color=TEXT)
    ax.set_ylabel("true class", color=TEXT)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or "Confusion matrix", loc="left", color=TEXT)
    fig.tight_layout()
    return fig
