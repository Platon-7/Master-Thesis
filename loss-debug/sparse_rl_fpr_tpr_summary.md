# Sparse-RL FPR/TPR comparison (CoffeePush, `reward_at_truncation=1`)

Each row = one reward model.
Setup: 60 clips per model (15 pre-success + 45 post-success per release demo).
GT = env reward at the clip's end frame (per-frame, non-sticky).

| Model | n_pos | n_neg | AUC | TPR@0%FPR | TPR@5%FPR | TPR@10%FPR | TPR@20%FPR | τ for TPR@0%FPR |
|---|---|---|---|---|---|---|---|---|
| Robometer-4B (post-fix baseline) | 116 | 184 | 0.653 | 0.043 | 0.060 | 0.112 | 0.181 | 0.063 |
| Robometer-FT step-3000 | 116 | 184 | 0.751 | 0.009 | 0.216 | 0.371 | 0.509 | 0.107 |
| Robometer-FT step-4000 | 116 | 184 | 0.663 | 0.000 | 0.000 | 0.000 | 0.155 | inf |
| Robometer-FT step-5000 | 116 | 184 | 0.668 | 0.017 | 0.103 | 0.138 | 0.319 | 0.172 |
| Qwen3.5-FT step-3000 | 116 | 184 | 0.801 | 0.000 | 0.043 | 0.328 | 0.664 | inf |
| Qwen3.5-FT step-4000 | 116 | 184 | 0.809 | 0.000 | 0.009 | 0.069 | 0.586 | inf |
| Qwen3.5-FT step-5000 | 116 | 184 | 0.780 | 0.000 | 0.078 | 0.284 | 0.586 | inf |
