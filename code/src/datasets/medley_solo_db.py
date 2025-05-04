from torch.utils.data import Dataset
import pandas as pd
from pydantic import BaseModel
import os
from typing import Tuple, Literal, Optional, List, Dict
import torch
from TorchJaekwon.Util.UtilAudio import UtilAudio

METADATA_CSV_DEFAULT_FILENAME = "Medley-solos-DB_metadata.csv"
MEDLEY_SOLODB_ID = "Medley-solos-DB"
WAVEFORM_FOLDER_DEFAULT_DIRNAME = MEDLEY_SOLODB_ID


class MedleySolosDBConstants:
    """
    Taken from https://zenodo.org/records/3464194
    """

    ORIGIN_SAMPLE_RATE: int = 44100
    ORIGIN_DURATION_APPROX: float = 2.972
    """The ground truth duration should be calculated by the original sample rate and the original number of samples."""
    MODE: Literal["stereo", "mono"] = "mono"
    BIT_DEPTH: int = 32
    ORIGIN_N_SAMPLES: int = 131072
    IDX_TO_DESCR: Dict[int, str] = {
        0: "clarinet",
        1: "distorted electric guitar",
        2: "female singer",
        3: "flute",
        4: "piano",
        5: "tenor saxophone",
        6: "trumpet",
        7: "violin",
        8: "unconditional",
    }
    TEST_SET_DISTRIBUTION: Dict[int, int] = {
        0: 732,
        1: 955,
        2: 1142,
        3: 3167,
        4: 2609,
        5: 325,
        6: 406,
        7: 2900,
    }
    GUITAR_IDX: int = 1


class MedleySolosDBDatasetConfig(BaseModel):
    """
    Either provide 'base_folder' or ('metadata_csv_path' and 'wavefile_folder_path')
    """

    base_folder: Optional[str] = None
    mode: Literal["training", "test", "validation", "-"] = "-"
    sample_rate: int
    mono: bool = True
    guitar_only: bool = False

    def set_mode(
        self, mode: Literal["training", "test", "validation"]
    ) -> "MedleySolosDBDatasetConfig":
        copy_ = self.model_copy()
        copy_.mode = mode
        return copy_

    def compute_paths(self) -> Tuple[str, str]:
        """
        Returns (metadata_csv_path, wavefile_folder_path)
        """
        metadata_csv_path = os.path.join(
            self.base_folder, METADATA_CSV_DEFAULT_FILENAME
        )
        waveform_folder_path = os.path.join(
            self.base_folder, WAVEFORM_FOLDER_DEFAULT_DIRNAME
        )
        return metadata_csv_path, waveform_folder_path


class MedleySolosDBDataset(Dataset):
    raw_metadata: pd.DataFrame
    metadata: pd.DataFrame

    def __init__(self, config: MedleySolosDBDatasetConfig) -> None:
        super().__init__()
        assert config.mode != "-"
        self.config = config
        self._load_metadata()
        if self.config.guitar_only:
            self.filter_instruments([MedleySolosDBConstants.GUITAR_IDX])

    def filter_instruments(self, instrument_labels: List[int]):
        """
        instrument_labels: indices of instruments to be kept
        """
        df = self.metadata.copy()
        df.loc[:, "instrument_id"] = df["instrument_id"].astype(int)

        self.metadata = df[df["instrument_id"].isin(instrument_labels)]

    def _load_metadata(self) -> pd.DataFrame:
        metadata_csv_path, _ = self.config.compute_paths()
        # load csv
        if not os.path.exists(metadata_csv_path):
            raise FileNotFoundError(
                f"Metadata CSV file not found at path: {metadata_csv_path}"
            )
        df = pd.read_csv(metadata_csv_path)

        # filter for training or testing or validation
        filtered_df = df[df["subset"] == self.config.mode]

        # set dataframes
        self.raw_metadata = df
        self.metadata = filtered_df

    @classmethod
    def from_basefolder(
        cls, basefolder: str, mode: Literal["training", "test", "validation"]
    ) -> "MedleySolosDBDataset":
        config = MedleySolosDBDatasetConfig(
            metadata_csv_path=os.path.join(basefolder, METADATA_CSV_DEFAULT_FILENAME),
            wavefile_folder_path=os.path.join(
                basefolder, WAVEFORM_FOLDER_DEFAULT_DIRNAME
            ),
            mode=mode,
        )
        return cls(config)

    def __uuid_to_filepath(self, uuid: str, label_idx: int) -> str:
        filepath = f"{MEDLEY_SOLODB_ID}_{self.config.mode}-{label_idx}_{uuid}.wav"
        return filepath

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns x,y where x is the audio and y is the label
        """
        _, waveform_folder_path = self.config.compute_paths()

        # load label
        y_index = self.metadata.iloc[index]["instrument_id"]
        y = torch.tensor([y_index])

        # load audio
        x_uuid = self.metadata.iloc[index]["uuid4"]
        x_fname = self.__uuid_to_filepath(x_uuid, label_idx=y_index)
        x_fpath = os.path.join(waveform_folder_path, x_fname)
        x, sample_rate = UtilAudio.read(
            audio_path=x_fpath,
            sample_rate=self.config.sample_rate,
            mono=True,
            module_name="torchaudio",
        )

        return x, y

    def __len__(self) -> int:
        return len(self.metadata)
