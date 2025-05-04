from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torchaudio.transforms as T
from pydantic import BaseModel


class MertConfig(BaseModel):
    embedding_idx: int = 0
    """Value in \{0, ..., 24\}"""


class Mert:
    def __init__(
        self, device, cfg: MertConfig = MertConfig(), sample_rate: int = 16000
    ):
        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.cfg = cfg
        self.model.to(device)
        self.device = device
        # loading the corresponding preprocessor config
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.sampling_rate = sample_rate
        self.resample_rate = self.processor.sampling_rate
        if self.resample_rate != self.sampling_rate:
            print(f"setting rate from {self.sampling_rate} to {self.resample_rate}")
            self.resampler = T.Resample(self.sampling_rate, self.resample_rate)
            self.resampler = self.resampler.to(self.device)
        else:
            self.resampler = None

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        if self.resampler is None:
            input_audio = x
        else:
            input_audio = self.resampler(x)

        inputs = self.processor(
            input_audio, sampling_rate=self.resample_rate, return_tensors="pt"
        )
        inputs["input_values"] = inputs["input_values"].squeeze().to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

        return time_reduced_hidden_states[self.cfg.embedding_idx]

    def sample_time_expanded(self, x: torch.Tensor) -> torch.Tensor:
        if self.resampler is None:
            input_audio = x
        else:
            input_audio = self.resampler(x)

        inputs = self.processor(
            input_audio, sampling_rate=self.resample_rate, return_tensors="pt"
        )
        inputs["input_values"] = inputs["input_values"].squeeze().to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        return all_layer_hidden_states[self.cfg.embedding_idx]
