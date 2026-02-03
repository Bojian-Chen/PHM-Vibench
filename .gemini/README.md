# STFT Pretrain + Few-shot Finetune (STFTTransformer)

This note documents the STFT-based pretrain + few-shot finetune pipeline.

## Run Commands

### 1) Pretrain only (single stage)
```bash
python main.py --config configs/experiments/pretrain_fewshot_transformer/stft_pretrain_only.yaml
```

### 2) Finetune only (single stage)
Provide the pretrained checkpoint via `model.weights_path`:
```bash
python main.py --config configs/experiments/pretrain_fewshot_transformer/stft_finetune_only.yaml \
  --override model.weights_path=/path/to/ckpt.ckpt
```

### 3) Full two-stage (pretrain → few-shot) in one run
```bash
python main.py --config configs/experiments/pretrain_fewshot_transformer/transformer_pretrain_fewshot.yaml
```

## Key Parameters

- `model.t_sec`: fixed physical duration (seconds) for each sample segment.
- `model.window_sec`: STFT window length (seconds).
- `model.hop_sec`: STFT hop length (seconds).
- `model.freq_max_hz`: optional frequency cutoff (Hz). If omitted, uses full band.
- `model.eps`: epsilon for `log(freq + eps)` to avoid nan/inf at f=0.
- `model.pos_mode`: positional encoding mode. Supported: `fixed_2d_sincos`.
- `model.patch_t`, `model.patch_f`: patch sizes in spectrogram time/freq bins.
- `model.channel_reduce`: how to handle multi-channel input (`first` or `mean`).

## Notes

- STFT window/hop are specified in seconds and converted to samples using each sample's true fs.
- Time coords use real seconds normalized to `[0,1]`.
- Frequency coords use `log(freq + eps)` normalized to `[0,1]` (no log-mel on amplitudes).
- The model uses fixed 2D sin-cos positional encoding on these real coords.

## Sanity Check
```bash
python scripts/stft_preprocess_sanity.py
```
