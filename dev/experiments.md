# SFT Experiments

Baseline for comparison: `sft_base_inherit` (no overrides; inherits defaults from the pretrained checkpoint / script defaults).

## Runs and Changes

| Run | Changed Hyperparameters | Description |
| --- | --- | --- |
| sft_base_inherit | None (all inherited defaults) | Reference run: baseline schedule and defaults inherited from the pretrained checkpoint. |
| sft_lr_half | embedding_lr=0.15, unembedding_lr=0.002, matrix_lr=0.01 | Tests stability and convergence with a conservative learning‑rate scale (half of base for all parameter groups). |
| sft_lr_1p5x | embedding_lr=0.45, unembedding_lr=0.006, matrix_lr=0.03 | Stress‑tests faster learning by increasing all learning rates 1.5x. |
| sft_lr_0p7x | embedding_lr=0.21, unembedding_lr=0.0028, matrix_lr=0.014 | Mildly conservative learning‑rate scale to see if slightly lower LR improves validation stability. |
| sft_lr_1p2x | embedding_lr=0.36, unembedding_lr=0.0048, matrix_lr=0.024 | Mildly aggressive learning‑rate scale to see if slightly higher LR improves speed without instability. |
| sft_warmup_05pct | warmup_ratio=0.05 | Faster ramp to full LR to test if shorter warmup improves early learning. |
| sft_warmup_10pct | warmup_ratio=0.10 | Slower ramp to full LR to test if longer warmup reduces early instability. |
| sft_warmdown_30pct | warmdown_ratio=0.30 | Earlier LR decay to test if stronger late‑stage regularization improves final quality. |
| sft_final_lr_01pct | final_lr_frac=0.01 | Very low LR floor to test if near‑zero final LR helps convergence. |
| sft_final_lr_05pct | final_lr_frac=0.05 | Moderate LR floor to keep learning active late in training. |
| sft_final_lr_10pct | final_lr_frac=0.10 | High LR floor to test if sustained learning late helps generalization. |
| sft_init_lr_100pct | init_lr_frac=1.0 | Starts at full base LR (no LR down‑scaling) to test sensitivity to initial LR. |
| sft_no_opt_warmstart | load_optimizer=0 | Fresh optimizer state to test whether warm‑starting momentum helps or hurts SFT. |
| sft_mmlu_heavy | mmlu_epochs=6, gsm8k_epochs=2 | Skews training mix toward MMLU to emphasize multiple‑choice reasoning over math. |
| sft_gsm8k_heavy | mmlu_epochs=2, gsm8k_epochs=8 | Skews training mix toward GSM8K to emphasize math/tool reasoning over multiple‑choice. |
