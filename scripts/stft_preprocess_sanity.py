"""Sanity check for STFT preprocessing + model forward/backward."""

from argparse import Namespace
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model_factory.Transformer.STFTTransformer import Model
from src.utils.training.masking import add_mask


def main():
    args = Namespace(
        d_model=32,
        n_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.1,
        num_classes=4,
        t_sec=0.05,
        window_sec=0.005,
        hop_sec=0.0025,
        eps=1e-6,
        pos_mode="fixed_2d_sincos",
        random_crop=False,
        stft_center=False,
        log_stft_params=False,
        channel_reduce="first",
        patch_t=4,
        patch_f=8,
        default_fs=12000,
    )

    model = Model(args, metadata=None)
    model.train()

    # Two samples with different fs
    fs_list = [12000.0, 48000.0]
    x1 = torch.randn(8192)
    x2 = torch.randn(8192)
    xs = [x1, x2]

    for idx, (x, fs) in enumerate(zip(xs, fs_list)):
        spec, meta = model.stft_preprocess_single(x, fs)
        time_axis = meta["time_axis_sec"]
        freq_axis = meta["freq_axis_hz"]

        # Coordinate ranges
        time_coord, freq_coord = model._build_coords(meta, 0, 0, spec.shape[0], spec.shape[1])
        assert torch.isfinite(time_coord).all(), "time_coord has inf/nan"
        assert torch.isfinite(freq_coord).all(), "freq_coord has inf/nan"
        assert time_coord.min() >= 0 and time_coord.max() <= 1 + 1e-6
        assert freq_coord.min() >= 0 and freq_coord.max() <= 1 + 1e-6

        # Mask + reconstruct
        masked_spec, total_mask = add_mask(spec.unsqueeze(0), forecast_part=0.1, mask_ratio=0.15)
        recon = model.reconstruct_from_spec(masked_spec.squeeze(0), meta)
        assert torch.isfinite(recon).all(), "reconstruction has inf/nan"

        mask = total_mask.squeeze(0)
        loss = ((recon - spec) ** 2)[mask].mean()
        loss.backward()
        print(f"Sample {idx}: spec={spec.shape}, time_bins={len(time_axis)}, freq_bins={len(freq_axis)}, loss={loss.item():.6f}")

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
