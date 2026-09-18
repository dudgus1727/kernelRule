# The autotuning curve (D-176 §1)

> **reproduce** `python3 -m experiments.autotune_curve --gpu <g> --fold <f>` then `python3 -m experiments.autotune_report --merge`
> 0 LLM calls · 0 GPU · the table is the measurement

| table | shapes | k=0 ours | k=0 vendor | random k=64 | pruned k=64 | cond-random k=64 | TPE k=64 | TPE dup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| a6000 | 65 | 1.114872 | 1.081488 | 1.0876 | 1.0919 | 1.1029 | 1.0836 | 60.6% |
| 5090 | 65 | 1.054384 | 1.145956 | 1.0290 | 1.0257 | 1.0416 | 1.0304 | 55.1% |
| 4090 | 63 | 1.053234 | 1.078236 | 1.0359 | 1.0369 | 1.0462 | 1.0321 | 56.7% |
| h100 | 63 | 1.068561 | 1.120733 | 1.1801 | 1.1722 | 1.1430 | 1.0912 | 57.5% |

## ★ Where autotuning catches us

| table | arm | vs our rule | vs vendor |
|---|---|---|---|
| a6000 | random | k=64 (interp 39.0) | k=128 (interp 76.1) |
| a6000 | random_pruned | k=64 (interp 41.4) | k=128 (interp 83.5) |
| a6000 | random_cond | k=64 (interp 51.9) | k=128 (interp 115.9) |
| a6000 | tpe | k=64 (interp 37.0) | k=128 (interp 68.6) |
| 5090 | random | k=32 (interp 24.0) | k=8 (interp 4.8) |
| 5090 | random_pruned | k=32 (interp 20.5) | k=8 (interp 4.3) |
| 5090 | random_cond | k=64 (interp 42.4) | k=8 (interp 6.4) |
| 5090 | tpe | k=32 (interp 29.9) | k=8 (interp 6.4) |
| 4090 | random | k=32 (interp 29.7) | k=16 (interp 15.6) |
| 4090 | random_pruned | k=32 (interp 30.4) | k=16 (interp 15.7) |
| 4090 | random_cond | k=64 (interp 49.3) | k=32 (interp 23.1) |
| 4090 | tpe | k=32 (interp 29.6) | k=16 (interp 14.5) |
| h100 | random | k=512 (interp 351.2) | k=256 (interp 132.0) |
| h100 | random_pruned | k=512 (interp 314.9) | k=128 (interp 126.1) |
| h100 | random_cond | k=256 (interp 198.6) | k=128 (interp 87.9) |
| h100 | tpe | k=128 (interp 98.0) | k=64 (interp 53.8) |

⚠️ `interp` is log-interpolated between the ks actually run (1, 2, 4, … 4096). It is not a measured k.

⚠️ **The TPE arm stops at k=1024**, the three random arms go to 4096. TPE's cost grows with both the duplicate rate and the number of completed trials (measured: 22.5 s at k=512, 163 s at k=1024, 1,065 s at k=2048 for one shape and one seed). It is not a cut where we win — TPE reaches regret 1.0000 by k=1024 on that shape and the crossing against our rule is near k=50. ⛔ The missing cells are left empty, not extrapolated. See `runs/x-autotune-aborted-k4096/`.

## ★ In seconds

| table | measure / candidate | build / candidate |
|---|--:|--:|
| a6000 | 0.026 s | 16.52 s |
| 5090 | 0.027 s | 14.38 s |
| 4090 | 0.030 s | 15.24 s |
| h100 | 0.025 s | 14.81 s |

★ Both numbers are the table's own: measure = `time_ms` × (`n_reps` + warmup) with the bundle `protocol` block's `warmup_frac` / `min_warmup`, build = `build_seconds`. ⛔ Nothing is estimated.
