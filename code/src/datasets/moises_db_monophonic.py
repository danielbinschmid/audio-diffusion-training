from moisesdb.dataset import MoisesDB
from torch.utils.data import Dataset
from pydantic import BaseModel
from typing import Tuple, Literal, List, Dict, Any, Callable
import os
import numpy as np
from moisesdb.track import MoisesDBTrack
from TorchJaekwon.Util.UtilAudio import UtilAudio
import pandas as pd
import torch
from tqdm import tqdm
from .utils import (
    get_start_end_idx,
    generate_windows,
    is_waveform_empty,
)
from schmid_werkzeug import print_info, print_warning


class MoisesDBConstants:
    ORIGIN_SAMPLE_RATE: int = 44100
    IS_MONO: bool = False
    GUITAR_STEM_SOURCE_LABEL_TO_IDX: Dict[str, int] = {
        "acoustic_guitar": 0,
        "clean_electric_guitar": 1,
        "distorted_electric_guitar": 2,
    }


class MoisesDBMonophonicConfig(BaseModel):
    ds_path: str = "/path/to/moisesdb"
    sample_rate: int = 16000
    mono: bool = True
    window_length_in_s: float = 6
    step_length_in_s: float = 2.5
    stem_mode: Literal["Guitar", "Bass"] = "Guitar"
    recompute_metadata: bool = False
    mode: Literal["train", "test", "-", "val"] = "-"
    train_val_test_split_seed: int = 1
    train_val_test_split_track_level: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    precomputed_train_metadata_path: str | None = None
    precomputed_test_metadata_path: str | None = None
    precomputed_val_metadata_path: str | None = None
    empty_waveform_thresshold: float = 1e-3
    exclude_label: List[int] = []

    def set_mode(
        self, mode: Literal["train", "test"], recompute_metadata: bool | None = None
    ) -> "MoisesDBMonophonicConfig":
        copy_ = self.model_copy()
        copy_.mode = mode
        if recompute_metadata is not None:
            copy_.recompute_metadata = recompute_metadata
        return copy_


class MoisesDBMonophonicDataset(Dataset):
    mode: Literal["train", "test", "-"]

    def __init__(self, cfg: MoisesDBMonophonicConfig, transforms: List[Callable] = []):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms
        self.window_size = (
            self.cfg.window_length_in_s * MoisesDBConstants.ORIGIN_SAMPLE_RATE
        )
        self.step_size = (
            self.cfg.step_length_in_s * MoisesDBConstants.ORIGIN_SAMPLE_RATE
        )

        if self.cfg.stem_mode != "Guitar" and self.cfg.stem_mode != "Bass":
            raise NotImplementedError()

        self.db = MoisesDB(
            data_path=self.cfg.ds_path,
            sample_rate=MoisesDBConstants.ORIGIN_SAMPLE_RATE,
            quiet=True,
        )

        self.mode = self.cfg.mode

        self.processed_subfolder_name = (
            f"preprocessed_extracted_{self.cfg.stem_mode}_monophonic"
        )

        if self.cfg.recompute_metadata:
            self.clear_metadata()

        self.preprocess()
        self.load_metadata()

    def preprocess(self) -> None:
        """
        Extract stems if not extracted yet and derive metadata if not derived yet.
        """
        if not os.path.exists(self.__get_stems_subfolder()):
            self._extract_stems()

        if not self._metadata_exists():
            self._serialise_metadata()

    def load_metadata(self) -> None:
        """
        Loads metadata csv file.
        """
        self.train_metadata = pd.read_csv(self.__get_metadata_path("train"))
        self.test_metadata = pd.read_csv(self.__get_metadata_path("test"))
        self.val_metadata = pd.read_csv(self.__get_metadata_path("val"))

    def save_metadata(self, t_folder: str) -> None:
        os.makedirs(t_folder, exist_ok=True)
        self.train_metadata.to_csv(os.path.join(t_folder, "train_metadata.csv"))
        self.test_metadata.to_csv(os.path.join(t_folder, "test_metadata.csv"))
        self.val_metadata.to_csv(os.path.join(t_folder, "val_metadata.csv"))

    def clear_metadata(self) -> None:
        """
        Removes metadata csv file and clears self.metadata pandas dataframe.
        """
        if os.path.exists(self.__get_metadata_path("train")):
            os.remove(self.__get_metadata_path("train"))
        if os.path.exists(self.__get_metadata_path("test")):
            os.remove(self.__get_metadata_path("test"))
        if os.path.exists(self.__get_metadata_path("val")):
            os.remove(self.__get_metadata_path("val"))
        self.train_metadata = None
        self.test_metadata = None
        self.val_metadata = None

    def clear_extracted_stems(self) -> None:
        """
        Removes directory with extracted stems.
        """
        print_warning("Cleaning extracted stems.")
        if os.path.exists(self.__get_stems_subfolder()):
            os.rmdir(self.__get_stems_subfolder())
        self.clear_metadata()

    def set_mode(self, mode: Literal["train", "test", "val"]):
        self.mode = mode

    def _extract_stems(self) -> None:
        """
        Extract stems via MoisesDB
        """
        print_info(f"Extracting {self.cfg.stem_mode} from MoisesDB dataset.")

        # create new subfolder where the stems should be saved to.
        new_subfolder_path = self.__get_stems_subfolder()
        if os.path.exists(new_subfolder_path):
            os.rmdir(new_subfolder_path)
        os.makedirs(new_subfolder_path)

        for track_idx in tqdm(range(len(self.db))):
            track: MoisesDBTrack = self.db[track_idx]
            stem_id: str = self._get_stem_id()
            if stem_id not in track.stems.keys():
                continue
            stem_sources: Dict[str, List[Dict[str, Any]]] = track.stem_sources(stem_id)

            for stem_key in stem_sources.keys():
                stem_source_list = stem_sources[stem_key]
                for i, stem_source in enumerate(stem_source_list):
                    stem_source_audio = stem_source["audio"]
                    stem_source_sample_rate = stem_source["sr"]
                    fpath = self.__get_stem_source_fpath(
                        track=track, stem_name=stem_key, stem_idx=i
                    )
                    UtilAudio.write(
                        audio_path=fpath,
                        audio=stem_source_audio,
                        sample_rate=stem_source_sample_rate,
                    )

    def _derive_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        generated metadata: |Path|Start|End|Label|

        Splits automatically into train and test split.

        Returns train_metadata, test_metadata
        """
        print_info("Deriving metadata for moises db dataset.")
        track_paths = [
            (os.path.join(self.__get_stems_subfolder(), track_filename), track_filename)
            for track_filename in os.listdir(self.__get_stems_subfolder())
        ]
        track_paths_filtered = [
            (path, name)
            for (path, name) in track_paths
            if MoisesDBConstants.GUITAR_STEM_SOURCE_LABEL_TO_IDX[
                self.__class_from_stem_fname(name)
            ]
            not in self.cfg.exclude_label
        ]
        print_info(
            f"Filtered {len(track_paths) - len(track_paths_filtered)} data samples."
        )
        track_paths = track_paths_filtered

        train_split, val_split, test_split = self.cfg.train_val_test_split_track_level
        assert train_split + val_split + test_split == 1
        n_tracks_train = int(train_split * len(track_paths))
        n_tracks_val = int(val_split * len(track_paths))
        print_info(
            f"Number of tracks for training: {n_tracks_train}, number of tracks for validation {n_tracks_val}, number of tracks for testing: {len(track_paths) - n_tracks_train- n_tracks_val}"
        )
        rnd_generator = np.random.default_rng(seed=self.cfg.train_val_test_split_seed)
        train_and_val_indeces = rnd_generator.choice(
            len(track_paths), n_tracks_train + n_tracks_val, replace=False
        )
        train_indices = train_and_val_indeces[:n_tracks_train]
        val_indices = train_and_val_indeces[n_tracks_train:]
        train_metadata = []
        test_metadata = []
        val_metadata = []
        for track_idx, (track_path, stem_fname) in enumerate(tqdm(track_paths)):
            class_label = self.__class_from_stem_fname(stem_fname)
            track, sample_rate = UtilAudio.read(
                track_path,
                sample_rate=MoisesDBConstants.ORIGIN_SAMPLE_RATE,
                module_name="torchaudio",
            )
            start, end = get_start_end_idx(
                track, threshold=self.cfg.empty_waveform_thresshold
            )
            if start is None:
                continue

            audio_waveform, sr = UtilAudio.read(
                track_path,
                sample_rate=MoisesDBConstants.ORIGIN_SAMPLE_RATE,
                mono=MoisesDBConstants.IS_MONO,
                start_idx=start,
                end_idx=end,
                module_name="torchaudio",
            )
            if is_waveform_empty(
                audio_waveform, threshold=self.cfg.empty_waveform_thresshold
            ):
                continue

            windows = generate_windows(
                start=start, end=end, window_size=self.window_size, step=self.step_size
            )
            for window in windows:
                start_sample, end_sample = window
                row = {
                    "Path": stem_fname,
                    "Start": start_sample,
                    "End": end_sample,
                    "Label": class_label,
                }
                if track_idx in train_indices:
                    train_metadata.append(row)
                elif track_idx in val_indices:
                    val_metadata.append(row)
                else:
                    test_metadata.append(row)

        return (
            pd.DataFrame(train_metadata),
            pd.DataFrame(val_metadata),
            pd.DataFrame(test_metadata),
        )

    def _get_metadata(self) -> pd.DataFrame:
        if self.mode == "train":
            assert self.train_metadata is not None
            return self.train_metadata
        elif self.mode == "test":
            assert self.test_metadata is not None
            return self.test_metadata
        elif self.mode == "val":
            assert self.val_metadata is not None
            return self.val_metadata
        else:
            return ValueError(f"Invalid mode '{self.mode}'")

    def _get_stem_id(self) -> str:
        """
        Get identifier for MoisesDB interface.
        """
        if self.cfg.stem_mode == "Guitar":
            return "guitar"
        elif self.cfg.stem_mode == "Bass":
            return "bass"
        else:
            raise NotImplementedError()

    def _serialise_metadata(self) -> None:
        """
        Saves metadata files (train and test) to csv files.
        """
        train_metadata, val_metadata, test_metadata = self._derive_metadata()
        train_metadata.to_csv(self.__get_metadata_path("train"))
        test_metadata.to_csv(self.__get_metadata_path("test"))
        val_metadata.to_csv(self.__get_metadata_path("val"))

    def _metadata_exists(self) -> bool:
        return (
            os.path.exists(self.__get_metadata_path("train"))
            and os.path.exists(self.__get_metadata_path("test"))
            and os.path.exists(self.__get_metadata_path("val"))
        )

    def __get_metadata_path(self, mode: Literal["train", "test", "val"]) -> str:

        if (
            self.cfg.precomputed_test_metadata_path is not None
            and self.cfg.precomputed_train_metadata_path is not None
            and self.cfg.precomputed_val_metadata_path is not None
        ):
            if mode == "train":
                metadata_path = self.cfg.precomputed_train_metadata_path
            elif mode == "val":
                metadata_path = self.cfg.precomputed_val_metadata_path
            else:
                metadata_path = self.cfg.precomputed_test_metadata_path

            return metadata_path
        return os.path.join(
            self.cfg.ds_path, f"{self.processed_subfolder_name}_{mode}_metadata.csv"
        )

    def __get_stems_subfolder(self) -> str:
        return os.path.join(self.cfg.ds_path, self.processed_subfolder_name)

    def __get_stem_source_fpath(
        self, track: MoisesDBTrack, stem_name: str, stem_idx: int
    ) -> str:
        return os.path.join(
            self.cfg.ds_path,
            self.processed_subfolder_name,
            f"{track.id}_class-{stem_idx}___{stem_name}.wav",
        )

    def __class_from_stem_fname(self, stem_fname: str) -> str:
        return stem_fname.rsplit("___", 1)[-1].removesuffix(".wav")

    def __len__(self):
        metadata = self._get_metadata()
        return metadata.shape[0]

    def __load_track(self, index: int, metadata: pd.DataFrame):
        track_metadata = metadata.iloc[index]
        trackfilename = track_metadata["Path"]
        start = track_metadata["Start"]
        end = track_metadata["End"]
        label = track_metadata["Label"]
        label_idx = MoisesDBConstants.GUITAR_STEM_SOURCE_LABEL_TO_IDX[label]

        trackpath = os.path.join(self.__get_stems_subfolder(), trackfilename)
        track, sr = UtilAudio.read(
            audio_path=trackpath,
            sample_rate=self.cfg.sample_rate,
            mono=self.cfg.mono,
            start_idx=start,
            end_idx=end,
            module_name="torchaudio",
        )
        x = track.unsqueeze(0)
        return x, label_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None]:
        metadata = self._get_metadata()

        # read track metadata
        x, label_idx = self.__load_track(index, metadata)
        for transform in self.transforms:
            x = transform(x)
        return x, torch.tensor([label_idx])
