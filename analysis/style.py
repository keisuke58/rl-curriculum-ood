"""Shared matplotlib style for publication-quality figures (Times-compatible)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

def apply():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         10,
        "axes.titlesize":    10,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.titlesize":  10,
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.5,
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":      42,  # embed fonts in PDF
        "ps.fonttype":       42,
    })
