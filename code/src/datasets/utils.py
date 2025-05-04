import torch
from typing import Tuple, List, Optional
import torch.nn.functional as F
from torch.utils.data import Sampler


def is_waveform_empty(waveform: torch.Tensor, threshold: float = 1e-4) -> bool:
    magnitude = waveform.abs().sum(dim=0)
    mean = magnitude.mean()
    return mean.item() < threshold


def get_start_end_idx(
    tensor: torch.Tensor, threshold: float = 1e-4, interval_size: int = 1000
) -> Tuple[Optional[int], Optional[int]]:
    """
    Identifies the start and end indices of non-zero regions in an interval-wise manner.

    Args:
        tensor (torch.Tensor): Input tensor of shape (2, n_samples).
        threshold (float): Threshold below which values are considered zero.
        interval_size (int): Number of samples per interval to check for threshold exceedance.

    Returns:
        Tuple[Optional[int], Optional[int]]: Start and end indices of the non-zero region.
    """

    # Compute the max across the two rows
    magnitude_max = tensor.max(dim=0)[0]

    # Split indices into intervals
    n_samples = magnitude_max.shape[0]
    interval_starts = torch.arange(0, n_samples, interval_size)

    # Find the first interval that contains a value above the threshold
    start_idx = None
    end_idx = None

    for i in interval_starts:
        interval_end = min(i + interval_size, n_samples)
        if (magnitude_max[i:interval_end].mean() > threshold).any():
            start_idx = i
            break

    # If no start index was found, return None
    if start_idx is None:
        return None, None

    # Find the last interval that contains a value above the threshold
    for i in reversed(interval_starts):
        interval_end = min(i + interval_size, n_samples)
        if (magnitude_max[i:interval_end] > threshold).any():
            end_idx = interval_end  # Use interval_end to include the last segment
            break

    return int(start_idx), int(end_idx)


def generate_windows(
    start: int, end: int, window_size: int, step: int
) -> List[Tuple[int, int]]:
    """
    Generate a list of (start, end) index pairs for a given range.
    
    :param start: Starting index of the range
    :param end: Ending index of the range
    :param window_size: Size of the window
    :param step: Step size to move the window
    :return: List of (start, end) tuples
    """
    windows = []
    current_start = start
    while current_start + window_size <= end:
        current_end = current_start + window_size
        windows.append((int(current_start), int(current_end)))
        current_start += step

    if windows and windows[-1][1] < end:
        last_start = max(end - window_size, start)  # Ensure it doesn't go below start
        windows.append((int(last_start), int(end)))

    return windows


def find_loudest_segment(
    waveform: torch.Tensor, sample_rate, duration=10, device="cuda", step_size=1
):
    window_size = sample_rate * duration

    # Square the waveform to get energy
    energy = waveform.to(device) ** 2

    # Use convolution for fast moving average
    window = torch.ones((int(window_size)), device=energy.device)
    energy_sums = F.conv1d(
        energy.view(1, 1, -1), window.view(1, 1, -1), stride=sample_rate * step_size
    ).squeeze()

    # Find the maximum energy segment
    max_idx = torch.argmax(energy_sums).item()
    start_sample = max_idx * sample_rate * step_size  # Corrected this line
    end_sample = start_sample + window_size

    return start_sample, end_sample


class RepeatLastSampleBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for idx in range(len(self.dataset)):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Repeat the last sample if needed
        if batch:
            while len(batch) < self.batch_size:
                batch.append(batch[-1])  # Repeat the last index
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
