"""The common subset used to compare two tables (different GPUs) (D-88).

When the 5090 table arrives, "does the structure transfer" has to be
measured. But **the two tables are not the same grid.**

    shape grid   66 vs 66, but the values differ (layer B's M raised, layer
                 E's ladder moved)
    config axes  8 vs 10 kinds of split_k
    ridge        159.1 vs 117.9  -> ★ the bound class of the same shape flips
    tick         1.024us vs 32ns

**Build the intersection explicitly, and count what cannot be controlled.**
Silently discarding what exists on only one side can make "it transferred" a
result of sample selection (§26.4).

## Config identity — decided by architecture-independent axes only

`kernel_id` compiles differently per architecture, so it cannot be a join key.
`regs_per_thread` / `smem_bytes` / `spill_bytes` are **build results** and
change with the GPU. What remains is only **the axes a human chooses**.

    ★ pipeline_kind is not included — the name may change between generations

⚠️ Even with this key equal it may be a **different kernel**. The same tile
axes use different instructions per generation. It is not "the same config"
but **"the same axis coordinates"**.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelrule.core.types import Hardware, Problem

__all__ = ["AXIS_FIELDS", "CrossReport", "axis_key", "common_shapes",
           "common_axis_keys", "bound_flipped", "cross_report"]

#: The **architecture-independent axes** that define config identity. Build
#: results (registers, smem, spills) and `kernel_id` change with the GPU, so
#: they are excluded.
AXIS_FIELDS = ("tile_m", "tile_n", "tile_k", "split_k", "split_k_mode",
               "align_a", "align_b", "align_c")


def axis_key(row) -> tuple:
    """One row's axis coordinates. `row` is a dict or a `Config`."""
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return tuple(get(f) for f in AXIS_FIELDS)


def common_shapes(a, b) -> list[Problem]:
    """Shapes present in both tables. **The order follows `a`**
    (deterministic)."""
    bk = {(p.M, p.N, p.K, p.dtype) for p in b.shapes()}
    return [p for p in a.shapes() if (p.M, p.N, p.K, p.dtype) in bk]


def common_axis_keys(a, b, p: Problem) -> set[tuple]:
    """Axis coordinates present in both tables for shape `p`."""
    return {axis_key(r) for r in a.frame_for(p).to_dict("records")} & \
           {axis_key(r) for r in b.frame_for(p).to_dict("records")}


def _arith_intensity(p: Problem) -> float:
    """FLOP / bytes moved. **A function of the shape alone** — no hardware.

    ## ★ Correction (2026-08-31) — the definition is not rewritten here

    This function used to multiply the output term by
    `acc_bytes_per_element` (f32, 4 bytes). But the accumulator lives in
    registers and **the C that goes out to DRAM is f16**. kernelTab's
    `arith_intensity` column and the registered feature
    `features.physical.arith_intensity` both count all three at element
    bytes.

    ```
    128x4096x4096   here 117.03   table/feature 120.471   5090 ridge 117.855
    ```

    **The boundary sits between them, so `bound_flipped` missed this
    shape** — it counted 4 flips as 3 in the 5090 transfer. The two
    definitions differed on **all** 53 common shapes.

    ★ Creating a third definition was the mistake (principle 2). It delegates
    to the registered feature.
    """
    from kernelrule.features.physical import arith_intensity

    return arith_intensity(p, None, None)


def _ridge(hw: Hardware) -> float:
    """★ It uses `hw.ridge_point` — the division is not redone here.

    Whether effective or spec values are used differs by 26% and flips the
    class of boundary shapes (§6.2). That judgement must live in `Hardware`
    alone.
    """
    return float(hw.ridge_point)


def bound_flipped(a, b, shapes=None) -> list[tuple[Problem, bool, bool]]:
    """Shapes whose **bound class flips** because of the ridge difference
    (D-88).

    `(shape, memory-bound in a?, memory-bound in b?)`.

    A rule may branch on `p.is_memory_bound`, so if the verdict differs
    per table **the two tables measure different things.** Letting it pass
    silently can make "it did not transfer" actually mean "we compared
    different things".

    ⛔ 2026-09-17 (D-179): this used to say "weights are fitted per regime".
    Scoring fits **one** vector now; what still depends on the verdict is
    the rule's own branch.
    """
    sh = shapes if shapes is not None else common_shapes(a, b)
    ra, rb = _ridge(a.hw), _ridge(b.hw)
    out = []
    for p in sh:
        ai = _arith_intensity(p)
        ma, mb = ai < ra, ai < rb
        if ma != mb:
            out.append((p, ma, mb))
    return out


@dataclass(frozen=True, slots=True)
class CrossReport:
    """The overlap of two tables. **It counts what was discarded.**"""

    n_shapes_a: int
    n_shapes_b: int
    n_shapes_common: int
    n_axis_a: int
    n_axis_b: int
    n_axis_common: int
    n_bound_flipped: int
    ridge_a: float
    ridge_b: float

    def render(self) -> str:
        def frac(k: int, n: int) -> str:
            return f"{k}/{n} = {k / n:.0%}" if n else f"{k}/0"
        drop_a = frac(self.n_shapes_a - self.n_shapes_common, self.n_shapes_a)
        drop_b = frac(self.n_shapes_b - self.n_shapes_common, self.n_shapes_b)
        flip = ("  — a rule branching on it measures different things in "
                "the two tables"
                if self.n_bound_flipped else "")
        return "\n".join([
            (f"  shapes    A {self.n_shapes_a}  B {self.n_shapes_b}  "
             f"common {self.n_shapes_common}"),
            f"            dropped from A {drop_a}   from B {drop_b}",
            (f"  axis coords A {self.n_axis_a}  B {self.n_axis_b}  "
             f"common {self.n_axis_common}"),
            (f"  ridge     A {self.ridge_a:.1f}  B {self.ridge_b:.1f}  "
             f"({self.ridge_b / self.ridge_a:.2f}x)"),
            f"  ★ bound flips  {self.n_bound_flipped} shapes{flip}",
        ])


def cross_report(a, b) -> CrossReport:
    sh = common_shapes(a, b)
    ka: set[tuple] = set()
    kb: set[tuple] = set()
    for p in a.shapes():
        ka |= {axis_key(r) for r in a.frame_for(p).to_dict("records")}
    for p in b.shapes():
        kb |= {axis_key(r) for r in b.frame_for(p).to_dict("records")}
    return CrossReport(
        n_shapes_a=len(a.shapes()), n_shapes_b=len(b.shapes()),
        n_shapes_common=len(sh), n_axis_a=len(ka), n_axis_b=len(kb),
        n_axis_common=len(ka & kb),
        n_bound_flipped=len(bound_flipped(a, b, sh)),
        ridge_a=_ridge(a.hw), ridge_b=_ridge(b.hw))
