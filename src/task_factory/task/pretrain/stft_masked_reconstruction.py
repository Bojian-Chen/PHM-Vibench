"""Masked reconstruction pretraining on STFT spectrograms."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from src.task_factory.Default_task import Default_task
from src.utils.training.masking import add_mask
from src.utils.metadata_utils import resolve_batch_metadata


class task(Default_task):
    """STFT masked reconstruction task (unsupervised)."""

    def __init__(
        self,
        network: nn.Module,
        args_data: Any,
        args_model: Any,
        args_task: Any,
        args_trainer: Any,
        args_environment: Any,
        metadata: Any,
    ):
        super().__init__(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata,
        )

        self.mask_ratio = getattr(args_task, "mask_ratio", 0.15)
        self.forecast_part = getattr(args_task, "forecast_part", 0.1)
        self.recon_loss = nn.MSELoss()

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        x = batch["x"].float()
        file_id = batch.get("file_id", None)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected input (B, L, C), got {x.shape}")

        B = x.shape[0]
        fs_tensor = resolve_batch_metadata(self.metadata, file_id, device=x.device)[1]

        losses = []
        for i in range(B):
            x_i = x[i]
            if x_i.shape[1] > 1:
                if getattr(self.network, "channel_reduce", "first") == "mean":
                    x_i = x_i.mean(dim=1)
                else:
                    x_i = x_i[:, 0]
            else:
                x_i = x_i[:, 0]
            spec, meta = self.network.stft_preprocess_single(x_i, float(fs_tensor[i].item()))
            masked_spec, total_mask = add_mask(spec.unsqueeze(0), self.forecast_part, self.mask_ratio)
            recon = self.network.reconstruct_from_spec(masked_spec.squeeze(0), meta)
            mask = total_mask.squeeze(0)
            if mask.any():
                loss_i = self.recon_loss(recon[mask], spec[mask])
            else:
                loss_i = (recon - spec).pow(2).mean()
            losses.append(loss_i)

        loss = torch.stack(losses).mean()
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_recon_loss": loss,
        }
        return metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, "train")
        self._log_metrics(metrics, "train")
        return metrics["train_loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics = self._shared_step(batch, "val")
        self._log_metrics(metrics, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics = self._shared_step(batch, "test")
        self._log_metrics(metrics, "test")
