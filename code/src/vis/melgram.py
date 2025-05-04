import numpy as np
import matplotlib.pyplot as plt


def vis_mel_gram(mel_gram: np.ndarray, fig_fpath: str):
    """
    mel_grams: assumed shape: (time, n_frequencies)
    """
    plt.figure(figsize=(7, 4), dpi=600)  # Compact figure size, high DPI for clarity
    plt.imshow(
        mel_gram.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest"
    )

    plt.axis("off")

    # Save the figure as a PDF with tight bounding box
    plt.savefig(fig_fpath, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the figure to free memory
