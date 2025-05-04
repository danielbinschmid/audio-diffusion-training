import numpy as np
import matplotlib.pyplot as plt


def array_to_figure(
    img_array: np.ndarray, save_path: str = "temp.png",
):
    """
        Convert a 1D ndarray into a line plot and save it as an image.

        Args:
            img_array (np.ndarray): The 1D data array.
            vmin (Optional[float]): Minimum y-axis range.
            vmax (Optional[float]): Maximum y-axis range.
            transposed (bool): Not applicable for 1D line plot.
            save_path (str): File path to save the image.
        """
    plt.figure(figsize=(8, 6))
    plt.plot(img_array, label="Line Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("1D Array Line Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
