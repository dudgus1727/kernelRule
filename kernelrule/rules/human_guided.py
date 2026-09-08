"""The baseline rule built from human guidance + table feedback (§9.4).

## ★ The name was fixed **twice** — the name was making a claim

```
handwritten      "a human wrote it"     ⛔ an LLM wrote it
physics_seeded   "it came from physics" ⛔ it was fixed four times against the
                                          table (1.776 -> 1.192)
★ human_guided   human guidance + table feedback + LLM        (D-128)
```

**The actual history** — written here so the name does not make a claim
again:

```
who wrote it   the LLM (Claude Code read the physics documents and wrote it
               in one go)
the human's    deciding the direction + ★ pointing out physical errors
share          **3 times**. It wrote no code
the table's    ★ four versions (1.776 -> 1.428 -> 1.221 -> 1.192). See the
share          "physics" section below
               ★ so it is **not** "built without looking at the table"
```

The difference from the evolved rule (`evolved`) is **not "who" but "what it
was fitted to, and how hard"** — this one to four rounds of summary
statistics, that one directly to the training split.

Several turns of conclusions accumulated on top of the two old names. See
`docs/artifacts/conclusion.md`.

**It is built before the LLM loop.** If this reaches regret 1.03, the LLM has
only 3% to fight over, which is about the size of the measurement noise — it
is the cheapest disproof experiment.

It is under **the same constraints** as a rule. The comparison is only fair
under the same conditions.

    the `score(f, p, hw, w)` signature
    registered features only
    numeric literals + len(W0) <= 8
    no branching on a config-level feature (`f.*` is an array)
    no direct comparison against a shape size

It is checked by `rules/checks.py`, and a test pins that fact.

## The physics — why these five terms

kernelTab's hand rule fixed its physics over four versions, and that history
is reflected here (`docs/baselines.md`). **Constants were not tuned; a wrong
model was fixed.**

    1.776  minimising tail_waste came first -> it picked a 32x128 tile with
           48 waves. ★ Worse than the static top-1
    1.428  a traffic term added. But it assumed tiles are always full -> 2.9
           at M=1
    1.221  traffic computed exactly as ceil(M/tm)*ceil(N/tn)*(tm+tn)
    1.192  SM utilisation 1/(1-tail_waste) added (a linear term cannot
           produce 5.3x)

The validation of our own feature library reproduced the same thing
independently:

    tail_waste standalone AUC = 0.176   <- it predicts **opposite** to the
                                           declared direction

Minimising `tail_waste` on its own selects small tiles (the smaller the tile,
the more tiles, so waves grows and tail_waste approaches 0). **It is
confounded with tile size**, so it only means something **together with** a
traffic term. That is why this rule puts traffic first and uses waves as a
correction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CODE", "W0", "score"]

#: The same kind of initial values an LLM proposes. The numerical optimiser
#: fits them (§29).
W0 = [1.0, 0.5, 0.4, 3.0, 0.3, 0.4, 0.5]

CODE = '''
def score(f, p, hw, w):
    s  = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.smem_pressure * w[2]
    s = s + f.has_spill * w[3]
    s = s + f.split_k_cost * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    if p.is_memory_bound:
        s = s + np.log2(f.traffic_amplification) * w[6]
    return s
'''


# ---------------------------------------------------------------------------
# ★ A shape-level branch only means something if it **reweights a
# config-level term**
# ---------------------------------------------------------------------------
# It was first written like this, and that was wrong:
#
#     if p.is_memory_bound:
#         s = s * w[2]          # ⛔ the ranking **does not change at all**
#
# Multiplying or adding a shape-level scalar into the whole score cannot
# change the ranking within that shape (it is a monotone transform). A rule
# is sorted independently per shape, so a shape constant cancels out.
#
# What a shape-level branch has to do is **change the relative weights of
# the config-level terms**. In the code above, that is the traffic term's
# weight going from `w[0]` to `w[0]+w[6]` when memory-bound.
#
# The static checks cannot catch this (it is syntactically legal). The
# scorer can see "does a shape-level branch change the ranking", so it
# belongs in the §12 diagnostic report.


def score(f, p, hw, w) -> np.ndarray:
    """Lower is better. Every term is dimensionless (§8.1 — the transfer
    premise).

    The order of the terms is the physical priority.

    1. `log2(traffic_amplification)` — the **log** of actual A/B DRAM
       traffic / the theoretical minimum. The main term. A larger tile means
       more reuse, so traffic falls. At M=1, `gm=1`, so raising tile_m does
       not reduce the tile count and only `(tm+tn)` grows, which becomes a
       penalty automatically — decode shapes need no special handling.

       ★ **Taking the log is the crux.** Used linearly this term runs away
       and picks tiles that minimise traffic but **spill**, such as
       256x256. In measurement, 256x256's rel median of 17.9 is the worst.
       The traffic gain saturates as the tile grows, so the log is also
       physically right.
    2. `sm_idle_cost` — the **non-linear** form of wave quantisation. Using
       128x128 on 512³ gives 16 tiles, so only 16 of 84 SMs run and the loss
       is 5.3x, which a linear `tail_waste=0.81` cannot produce.
    3. `smem_pressure` — filling smem reduces the resident blocks per SM.
    4. `has_spill` — a spilling kernel was picked as optimal 0 times in this
       table (rel median 13.6). A binary penalty is enough.
    5. `split_k_cost` — the cost of round-tripping D per partition.
    6. `pipeline_warmup_frac` — a deep pipeline on a short mainloop cannot
       pay back the cost of filling it.

    On memory-bound shapes it **raises the weight of the traffic term.**
    Once at the bandwidth floor, bytes dominate over compute efficiency. It
    generalises because it says "once at the bandwidth floor", not "when M
    is 4096".
    """
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.smem_pressure * w[2]
    s = s + f.has_spill * w[3]
    s = s + f.split_k_cost * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    if p.is_memory_bound:
        s = s + np.log2(f.traffic_amplification) * w[6]
    return s
