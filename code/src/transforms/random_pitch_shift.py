import torchaudio.functional as a_F
import random


class RandomPitchShift:
    def __init__(
        self,
        sample_rate: int,
        min_n_steps: int = -5,
        max_n_steps: int = 5,
        bins_per_octave: int = 12,
        p: float = 0.5,
    ):

        self.sample_rate = sample_rate
        self.min_n_steps = min_n_steps
        self.max_n_steps = max_n_steps
        self.bins_per_octave = bins_per_octave
        self.p = p

    def __call__(self, waveform):
        """
        Apply random pitch shifting to the given waveform.

        Args:
            waveform (Tensor): Audio waveform of shape (channels, time)

        Returns:
            Tensor: Augmented waveform with pitch shift applied.
        """
        if random.random() < self.p:  # Apply transform with probability p
            n_steps = random.uniform(self.min_n_steps, self.max_n_steps)
            return a_F.pitch_shift(
                waveform,
                self.sample_rate,
                n_steps,
                bins_per_octave=self.bins_per_octave,
            )
        return waveform  # Return original waveform if not applying transform
