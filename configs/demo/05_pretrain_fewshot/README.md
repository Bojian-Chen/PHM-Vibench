# Demo: Pretrain + Few-shot (`demo_05_pretrain_fewshot`)

## Purpose

Pipeline_02 pretrain/few-shot pipeline: now runs as a **true two-stage** config via `stages:`
(`pretrain` → `fewshot_finetune`).

## Minimal Run

```bash
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# Stage-specific overrides (optional)
# --override stages[0].trainer.num_epochs=1 --override stages[1].trainer.num_epochs=1
```

## Common Pitfalls

1) Forgetting that stage overrides inherit global config (set stage-specific overrides if needed).
2) Confusing paper-only pipeline03 scripts with this repo demo.
