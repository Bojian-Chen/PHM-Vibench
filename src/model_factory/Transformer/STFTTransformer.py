"""STFT-based Transformer with 2D coordinate positional encoding.

This model performs:
- fixed-duration cropping in seconds
- STFT using window/hop in seconds
- 2D patch embedding on the spectrogram
- fixed 2D sin-cos positional encoding from real time/freq coordinates
- Transformer encoder for representation learning

It supports:
- reconstruction from masked spectrograms (for pretraining)
- classification head (for finetune)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.metadata_utils import resolve_batch_metadata


class Model(nn.Module):
    def __init__(self, args, metadata: Any = None):
        super().__init__()
        self.metadata = metadata

        # Transformer config
        self.d_model = int(getattr(args, "d_model", 128))
        self.n_heads = int(getattr(args, "n_heads", 8))
        self.num_layers = int(getattr(args, "num_layers", 4))
        self.d_ff = int(getattr(args, "d_ff", 256))
        self.dropout = float(getattr(args, "dropout", 0.1))
        self._warned_multi_classes = False
        raw_num_classes = getattr(args, "num_classes", None)
        self.num_classes = self._resolve_num_classes(raw_num_classes)

        # STFT config (seconds)
        self.t_sec = float(getattr(args, "t_sec", 0.05))
        self.window_sec = float(getattr(args, "window_sec", 0.005))
        self.hop_sec = float(getattr(args, "hop_sec", 0.0025))
        self.freq_max_hz = getattr(args, "freq_max_hz", None)
        self.eps = float(getattr(args, "eps", 1e-6))
        self.pos_mode = str(getattr(args, "pos_mode", "fixed_2d_sincos"))
        self.random_crop = bool(getattr(args, "random_crop", False))
        self.center = bool(getattr(args, "stft_center", False))
        self.log_stft_params = bool(getattr(args, "log_stft_params", True))
        self.default_fs = getattr(args, "default_fs", None)
        self.channel_reduce = str(getattr(args, "channel_reduce", "first"))

        # Patch config (spectrogram domain)
        self.patch_t = int(getattr(args, "patch_t", 8))
        self.patch_f = int(getattr(args, "patch_f", 8))

        self.patch_proj = nn.Linear(self.patch_t * self.patch_f, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.recon_head = nn.Linear(self.d_model, self.patch_t * self.patch_f)
        self.classifier = nn.Linear(self.d_model, self.num_classes) if self.num_classes is not None else None

        self._logged_stft = set()
        self._warned_channel_reduce = False

    # ------------------------- public API -------------------------
    def stft_preprocess_single(self, x_1d: torch.Tensor, fs: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute STFT spectrogram for a single waveform.

        Returns:
            spec: (T, F) magnitude spectrogram
            meta: dict with time_axis_sec, freq_axis_hz, window_samples, hop_samples, n_fft, fs
        """
        if x_1d.dim() != 1:
            raise ValueError(f"Expected 1D waveform, got shape {x_1d.shape}")

        fs_val = float(fs)
        target_len = max(1, int(round(fs_val * self.t_sec)))
        if x_1d.numel() >= target_len:
            if self.random_crop and x_1d.numel() > target_len:
                start = int(torch.randint(0, x_1d.numel() - target_len + 1, (1,)).item())
            else:
                start = 0
            x_seg = x_1d[start:start + target_len]
        else:
            pad = target_len - x_1d.numel()
            x_seg = F.pad(x_1d, (0, pad))

        window_samples = max(2, int(round(fs_val * self.window_sec)))
        hop_samples = max(1, int(round(fs_val * self.hop_sec)))
        n_fft = window_samples

        if self.log_stft_params:
            key = (int(round(fs_val)), window_samples, hop_samples)
            if key not in self._logged_stft:
                print(
                    f"[STFT] fs={fs_val:.2f}Hz window_sec={self.window_sec} hop_sec={self.hop_sec} "
                    f"-> window_samples={window_samples} hop_samples={hop_samples}"
                )
                self._logged_stft.add(key)

        window = torch.hann_window(window_samples, device=x_1d.device)
        stft = torch.stft(
            x_seg,
            n_fft=n_fft,
            hop_length=hop_samples,
            win_length=window_samples,
            window=window,
            center=self.center,
            return_complex=True,
        )
        spec = torch.abs(stft).transpose(0, 1)  # (T, F)

        time_axis_sec = torch.arange(spec.shape[0], device=x_1d.device, dtype=torch.float32) * (hop_samples / fs_val)
        freq_axis_hz = torch.linspace(0.0, fs_val / 2.0, spec.shape[1], device=x_1d.device)

        if self.freq_max_hz is not None:
            freq_max = float(self.freq_max_hz)
            mask = freq_axis_hz <= freq_max
            if mask.any():
                spec = spec[:, mask]
                freq_axis_hz = freq_axis_hz[mask]

        meta = {
            "time_axis_sec": time_axis_sec,
            "freq_axis_hz": freq_axis_hz,
            "window_samples": window_samples,
            "hop_samples": hop_samples,
            "n_fft": n_fft,
            "fs": fs_val,
        }
        return spec, meta

    def reconstruct_from_spec(self, spec: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct spectrogram from (masked) input spec."""
        tokens, patch_meta = self._spec_to_tokens(spec, meta)
        encoded = self.encoder(tokens.unsqueeze(0))
        pred_patches = self.recon_head(encoded.squeeze(0))
        recon = self._unpatchify(pred_patches, patch_meta)
        return recon

    def forward(self, x: torch.Tensor, data_id: Any = None, task_id: Optional[str] = None) -> torch.Tensor:
        """Classification forward. Computes STFT internally.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform (B, L, C) or (L, C) or (L,).
        data_id : Any
            file_id or list/tensor of ids for metadata lookup.
        task_id : str, optional
            Only 'classification' is supported in forward.
        """
        if task_id is None:
            task_id = "classification"
        if task_id != "classification":
            raise ValueError(f"Unsupported task_id for forward: {task_id}")
        if self.classifier is None:
            raise RuntimeError("Classifier head is not initialized (num_classes is None).")

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, L, C), got {x.shape}")

        x = x.float()
        B, _, C = x.shape
        if C > 1:
            if self.channel_reduce == "mean":
                x = x.mean(dim=2, keepdim=True)
            else:
                x = x[:, :, :1]
            if not self._warned_channel_reduce:
                print(f"[STFT] channel_reduce={self.channel_reduce} (C={C} -> 1)")
                self._warned_channel_reduce = True

        fs_tensor = self._resolve_fs(data_id, batch_size=B, device=x.device)
        logits = []
        for i in range(B):
            logit_i = self._forward_single_classification(x[i, :, 0], float(fs_tensor[i].item()))
            logits.append(logit_i)
        return torch.stack(logits, dim=0)

    # ------------------------- internals -------------------------
    def _resolve_fs(self, data_id: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.metadata is not None and data_id is not None:
            _, fs_tensor = resolve_batch_metadata(self.metadata, data_id, device)
            return fs_tensor
        if self.default_fs is None:
            raise ValueError("default_fs must be provided when metadata/data_id is unavailable.")
        return torch.full((batch_size,), float(self.default_fs), device=device)

    def _resolve_num_classes(self, num_classes: Any) -> Optional[int]:
        if num_classes is None:
            return None
        if isinstance(num_classes, dict):
            if len(num_classes) == 0:
                return None
            if len(num_classes) == 1:
                return int(next(iter(num_classes.values())))
            if not self._warned_multi_classes:
                print(
                    "[STFT] num_classes contains multiple datasets; "
                    "classification head disabled (pretrain-only usage)."
                )
                self._warned_multi_classes = True
            return None
        try:
            return int(num_classes)
        except (TypeError, ValueError):
            return None

    def _forward_single_classification(self, x_1d: torch.Tensor, fs: float) -> torch.Tensor:
        spec, meta = self.stft_preprocess_single(x_1d, fs)
        tokens, _ = self._spec_to_tokens(spec, meta)
        encoded = self.encoder(tokens.unsqueeze(0))
        pooled = encoded.mean(dim=1).squeeze(0)
        return self.classifier(pooled)

    def _spec_to_tokens(self, spec: torch.Tensor, meta: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if spec.dim() != 2:
            raise ValueError(f"Expected spec shape (T, F), got {spec.shape}")

        spec_padded, pad_t, pad_f = self._pad_spec(spec)
        T_pad, F_pad = spec_padded.shape
        t_patches = T_pad // self.patch_t
        f_patches = F_pad // self.patch_f

        patches = spec_padded.unfold(0, self.patch_t, self.patch_t).unfold(1, self.patch_f, self.patch_f)
        patches = patches.contiguous().view(t_patches * f_patches, self.patch_t * self.patch_f)

        time_coord, freq_coord = self._build_coords(meta, pad_t, pad_f, T_pad, F_pad)
        pos = self._build_pos_embed(time_coord, freq_coord, t_patches, f_patches, device=spec.device)

        tokens = self.patch_proj(patches) + pos
        patch_meta = {
            "orig_T": spec.shape[0],
            "orig_F": spec.shape[1],
            "T_pad": T_pad,
            "F_pad": F_pad,
            "t_patches": t_patches,
            "f_patches": f_patches,
        }
        return tokens, patch_meta

    def _pad_spec(self, spec: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        t_dim, f_dim = spec.shape
        pad_t = (self.patch_t - (t_dim % self.patch_t)) % self.patch_t
        pad_f = (self.patch_f - (f_dim % self.patch_f)) % self.patch_f
        if pad_t > 0 or pad_f > 0:
            spec = F.pad(spec, (0, pad_f, 0, pad_t))
        return spec, pad_t, pad_f

    def _build_coords(self, meta: Dict[str, Any], pad_t: int, pad_f: int, T_pad: int, F_pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
        time_axis_sec = meta["time_axis_sec"]
        freq_axis_hz = meta["freq_axis_hz"]
        fs_val = float(meta["fs"])
        hop_samples = int(meta["hop_samples"])
        n_fft = int(meta["n_fft"])

        if pad_t > 0:
            step_t = hop_samples / fs_val
            extra_t = time_axis_sec[-1] + step_t * torch.arange(1, pad_t + 1, device=time_axis_sec.device)
            time_axis_sec = torch.cat([time_axis_sec, extra_t], dim=0)
        if pad_f > 0:
            step_f = fs_val / n_fft
            extra_f = freq_axis_hz[-1] + step_f * torch.arange(1, pad_f + 1, device=freq_axis_hz.device)
            freq_axis_hz = torch.cat([freq_axis_hz, extra_f], dim=0)

        time_axis_sec = time_axis_sec[:T_pad]
        freq_axis_hz = freq_axis_hz[:F_pad]

        time_max = time_axis_sec[-1] if time_axis_sec.numel() > 0 else torch.tensor(1.0, device=time_axis_sec.device)
        if time_max.item() <= 0:
            time_coord = torch.zeros_like(time_axis_sec)
        else:
            time_coord = time_axis_sec / (time_max + self.eps)

        freq_log = torch.log(freq_axis_hz + self.eps)
        f_min = freq_log.min() if freq_log.numel() > 0 else torch.tensor(0.0, device=freq_log.device)
        f_max = freq_log.max() if freq_log.numel() > 0 else torch.tensor(1.0, device=freq_log.device)
        denom = (f_max - f_min).clamp(min=self.eps)
        freq_coord = (freq_log - f_min) / denom

        return time_coord, freq_coord

    def _build_pos_embed(self, time_coord: torch.Tensor, freq_coord: torch.Tensor, t_patches: int, f_patches: int, device: torch.device) -> torch.Tensor:
        if self.pos_mode != "fixed_2d_sincos":
            raise ValueError(f"Unsupported pos_mode: {self.pos_mode}")

        time_coord = time_coord.view(t_patches, self.patch_t).mean(dim=1)
        freq_coord = freq_coord.view(f_patches, self.patch_f).mean(dim=1)

        grid_t, grid_f = torch.meshgrid(time_coord, freq_coord, indexing="ij")
        t_flat = grid_t.reshape(-1).to(device)
        f_flat = grid_f.reshape(-1).to(device)

        dim_t = self.d_model // 2
        dim_f = self.d_model - dim_t
        emb_t = self._sincos_1d(t_flat, dim_t)
        emb_f = self._sincos_1d(f_flat, dim_f)
        return torch.cat([emb_t, emb_f], dim=1)

    def _sincos_1d(self, coord: torch.Tensor, dim: int) -> torch.Tensor:
        if dim <= 0:
            return coord.new_zeros((coord.shape[0], 0))
        half = dim // 2
        if half == 0:
            return coord.new_zeros((coord.shape[0], dim))
        omega = torch.arange(half, device=coord.device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / half))
        out = coord[:, None] * omega[None, :]
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _unpatchify(self, pred_patches: torch.Tensor, patch_meta: Dict[str, Any]) -> torch.Tensor:
        t_patches = patch_meta["t_patches"]
        f_patches = patch_meta["f_patches"]
        T_pad = patch_meta["T_pad"]
        F_pad = patch_meta["F_pad"]
        orig_T = patch_meta["orig_T"]
        orig_F = patch_meta["orig_F"]

        patches = pred_patches.view(t_patches, f_patches, self.patch_t, self.patch_f)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        spec = patches.view(T_pad, F_pad)
        return spec[:orig_T, :orig_F]
