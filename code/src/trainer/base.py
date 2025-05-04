import uuid
from schmid_werkzeug import print_info
import os


class BaseTrainer:
    ckp_path: str
    """Path where intermediate checkpoints are saved."""

    def __init__(self, ckpt_path: str, skip_creating_ckpt_dir: bool = False):
        if skip_creating_ckpt_dir:
            self.ckp_path = ""
        else:
            self.set_ckp_path(ckp_path=ckpt_path)

    def set_ckp_path(self, ckp_path: str):
        self._uuid = uuid.uuid4()
        self.ckp_path = os.path.join(ckp_path, f"{self._uuid}")
        print_info(f"Saving checkpoints to {self.ckp_path}")
        os.makedirs(self.ckp_path, exist_ok=True)
