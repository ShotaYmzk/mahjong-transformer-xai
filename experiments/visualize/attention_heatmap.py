"""Attention heatmap export."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_attention_heatmap(attention, output_path: str | Path, *, title: str = "Attention") -> None:
    """Save an attention matrix as PNG when matplotlib is available, else CSV."""
    arr = attention.detach().cpu().numpy() if hasattr(attention, "detach") else np.asarray(attention)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(arr, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
    except Exception:
        np.savetxt(output.with_suffix(".csv"), arr, delimiter=",")
