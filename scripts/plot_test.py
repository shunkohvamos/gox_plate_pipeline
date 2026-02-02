"""Test script to verify paper-grade plot styling."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import paper-grade style
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gox_plate_pipeline.fitting.core import apply_paper_style, PAPER_FIGSIZE_SINGLE

out = Path("out")
out.mkdir(exist_ok=True)

x = np.linspace(0, 10, 100)
y = np.sin(x)

with plt.rc_context(apply_paper_style()):
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
    ax.plot(x, y, color="#0072B2")
    ax.set_title("Example plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "test.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

print("saved:", out / "test.png")
