from pydantic import BaseModel
from src.logger import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM
from typing import Literal, Optional, List, Callable
from tqdm import tqdm
import torch
import os
import sys
from schmid_werkzeug import print_info
from .base import BaseTrainer
from src.models.guided_diffusion_modules.GuidedDiffusionUnet import GuidedDiffusionUnet
import torch.nn.functional as F


class DiffusionTrainerConfig(BaseModel):
    device: str = "cuda"
    num_epochs: int = 2000
    num_workers_dataloader: int = 16
    batch_size: int = 16
    checkpoint_path: str
    repa_gamma: float = 1.0


class DiffusionTrainer(BaseTrainer):
    """
    Assumes that dataset returns tuple (x,y) where x is audio and y is label.
    """

    def __init__(
        self,
        trainer_cfg: DiffusionTrainerConfig,
        logger: WandbLogger,
        train_ds: Dataset,
        scheduler: LRScheduler,
        ddpm: DDPM,
        val_ds: Optional[Dataset] = None,
        test_ds: Optional[Dataset] = None,
        repa_embedding_sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        repa_align: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = trainer_cfg
        self.logger = logger
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.scheduler = scheduler
        self.ddpm = ddpm
        self.repa_align = repa_align
        self.init_dataloaders()
        self.repa_emb_sampler = repa_embedding_sampler
        super().__init__(ckpt_path=trainer_cfg.checkpoint_path)

        self._on_epoch_end: List[Callable[[int, DDPM, WandbLogger], None]] = []

    def register_hook(
        self,
        hook_type: Literal["on_epoch_end"],
        hook: Callable[[int, DDPM, WandbLogger], None],
    ) -> None:
        """
        on_epoch_end: Arguments (epoch_idx: int, ddpm: DDPM, logger: WandbLogger)
        """
        if hook_type != "on_epoch_end":
            raise NotImplementedError("Only on_epoch_end hook implemented")

        self._on_epoch_end.append(hook)

    def init_dataloaders(self) -> None:
        self.dataloader_train = DataLoader(
            dataset=self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers_dataloader,
        )
        self.dataloader_test = (
            DataLoader(
                dataset=self.test_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers_dataloader,
            )
            if self.test_ds is not None
            else None
        )
        self.dataloader_val = (
            DataLoader(
                dataset=self.val_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers_dataloader,
            )
            if self.val_ds is not None
            else None
        )

    def load_ddpm_ckpt(self, ckpt_path: str, device="cpu") -> None:
        state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
        self.ddpm.load_state_dict(state_dict)

    def fit_ddpm(self, from_epoch: int = 0) -> None:
        assert from_epoch < self.config.num_epochs

        if from_epoch > 0:
            print_info(f"Resuming from epoch {from_epoch}")

        for i in range(from_epoch):
            self.scheduler.step()

        self.ddpm = self.ddpm.to(self.config.device)

        best_checkpoint_loss = sys.float_info.max

        for epoch in range(from_epoch, self.config.num_epochs):
            self.ddpm.train()
            running_loss = 0.0

            for batch_idx, (x, y) in enumerate(tqdm(self.dataloader_train)):

                # map to execution device
                x: torch.Tensor = x.to(self.config.device)
                y: torch.Tensor = y.to(self.config.device)

                # compute loss
                condition = {"class_label": y.to(x.device)}

                if self.repa_emb_sampler is not None:
                    condition["cache_intermediate_activations"] = True
                    unet: GuidedDiffusionUnet = self.ddpm.model
                    unet.intermediate_activation_cache.clear()
                    unet.intermediate_activation_cache.set_cuda()

                diffusion_loss: torch.Tensor = self.ddpm(
                    x_start=x, cond=condition, is_cond_unpack=True
                )
                raw_diff_loss = diffusion_loss.detach().cpu().item()

                if self.repa_emb_sampler is not None:
                    unet: GuidedDiffusionUnet = self.ddpm.model
                    t_key = unet.intermediate_activation_cache.time_keys()
                    assert len(t_key) == 1
                    hs: torch.Tensor = unet.intermediate_activation_cache.get_hspace(
                        t_key[0]
                    )[0]
                    hs_permuted = hs.permute(0, 3, 1, 2)
                    shape = hs_permuted.shape[0], hs_permuted.shape[1], -1
                    hs_processed = hs_permuted.view(shape)  # (bs, t, 1024)

                    emb = self.repa_emb_sampler(x)

                    t = emb.shape[1]
                    t2 = hs_permuted.shape[1]
                    a = emb

                    kernel_size = t / t2  # This must be an integer in an ideal case

                    if kernel_size.is_integer():
                        kernel_size = int(kernel_size)
                        emb_resized = F.avg_pool1d(
                            a.permute(0, 2, 1), kernel_size, stride=kernel_size
                        ).permute(0, 2, 1)
                    else:
                        # Use interpolation for non-integer downsampling
                        emb_resized = F.interpolate(
                            a.permute(0, 2, 1),
                            size=t2,
                            mode="linear",
                            align_corners=False,
                        ).permute(0, 2, 1)

                    hs_ = hs_processed
                    hs_shape = hs_.shape
                    hs_ = self.repa_align(hs_.reshape((-1, hs_shape[2]))).reshape(
                        hs_shape
                    )

                    reg_loss = torch.tensor(0.0, device=self.config.device)
                    for i in range(t2):
                        reg_loss = reg_loss + F.cosine_embedding_loss(
                            input1=emb_resized[:, i],
                            input2=hs_[:, i].to(self.config.device),
                            target=torch.ones(
                                (emb_resized.shape[0]), device=self.config.device
                            ),
                        )
                    reg_loss = reg_loss / t2
                    self.logger.log_custom_float("REPA", reg_loss.detach().cpu().item())
                    diffusion_loss = diffusion_loss + self.config.repa_gamma * reg_loss

                # backprop
                self.scheduler.optimizer.zero_grad()
                diffusion_loss.backward()
                self.scheduler.optimizer.step()

                # for logging
                running_loss += diffusion_loss.item()
                self.logger.log_loss(raw_diff_loss)

            # scheduler step
            self.scheduler.step()

            # on epoch end save ckp-point and call hooks
            epoch_loss = running_loss / len(self.dataloader_train)
            self.logger.log_epoch_avg_loss(
                epoch_loss, lr=self.scheduler.get_last_lr()[0]
            )  # assume only one parameter group for lr
            torch.save(
                self.ddpm.state_dict(), os.path.join(self.ckp_path, f"{epoch}.pth")
            )
            if self.repa_align is not None:
                torch.save(
                    self.repa_align.state_dict(),
                    os.path.join(self.ckp_path, f"repa_align_{epoch}.pth"),
                )

            # save best checkpoint
            if epoch_loss < best_checkpoint_loss:
                torch.save(
                    self.ddpm.state_dict(),
                    os.path.join(self.ckp_path, f"best_ckpt_{epoch_loss}_{epoch}.pth"),
                )
                if self.repa_align is not None:
                    torch.save(
                        self.repa_align.state_dict(),
                        os.path.join(
                            self.ckp_path,
                            f"repa_align_best_ckpt_{epoch_loss}_{epoch}.pth",
                        ),
                    )

                best_checkpoint_loss = epoch_loss

            for epoch_end_hook in self._on_epoch_end:
                epoch_end_hook(epoch, self.ddpm, self.logger)
            print_info(
                f"Epoch [{epoch+1}/{self.config.num_epochs}], Average Loss: {epoch_loss:.4f}"
            )
