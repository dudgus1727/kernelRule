# The single-agent wording, in full (D-176 §2-2 · §4)

> **reproduce** the block is `kernelrule/agents/prompts/role/_size_single.md`;
> the assembled prompt below is the one **actually sent**, taken from
> `runs/sa-a6000-f0/stage2-rule-writer/llm_calls/` — not re-rendered here.

⚠️ **The loop's wording (`d08a48d`) is untouched.** The two differ in exactly
one block: `rule_writer.md` carries a `{size_guidance_block}` placeholder and
`--size-guidance` chooses which file fills it. The diff below is the whole
difference between what campaign 2's seeds were asked and what this control
was asked.

## The block

```markdown
**Write the best rule you can.**

⚠️ This variant is the **single-agent control** (D-176 §2). No evolution
loop follows it: whatever this rule is, that is the answer. Do not hold
anything back for a later round.

**Branch where the bottleneck differs.** A memory-bound shape and a
compute-bound shape are not ranked by the same weights, so a split on a
shape-level value (`p.<name>` above) belongs in the rule.
```

## The whole difference against `d08a48d`

```diff
--- d08a48d (loop seed)
+++ D-176 §2-2 (single agent)
@@ -71,23 +71,15 @@
     return s
 ```
 
-That is the smallest honest shape.
+**Write the best rule you can.**
 
-**The bar for adding a term is not "I can name the physics" — it is "leaving
-it out makes the ranking obviously worse."** You can name the physics of
-dozens of terms; that is not the question. Put in the ones whose absence you
-could point to in a specific candidate, and stop.
+⚠️ This variant is the **single-agent control** (D-176 §2). No evolution
+loop follows it: whatever this rule is, that is the answer. Do not hold
+anything back for a later round.
 
-★ **The rest is not your job.** This rule is a seed: an evolution loop then
-reads the cases it gets wrong and adds terms for them, one at a time, with
-evidence. A term you add here on a hunch takes that slot away — it arrives
-without a case behind it and the loop has to spend rounds discovering it was
-never needed.
-
-**Split when the bottleneck differs.** A memory-bound shape and a
-compute-bound shape are not ranked by the same weights, so a branch on a
-shape-level value (`p.<name>` above) is part of the smallest honest shape,
-not an addition to it.
+**Branch where the bottleneck differs.** A memory-bound shape and a
+compute-bound shape are not ranked by the same weights, so a split on a
+shape-level value (`p.<name>` above) belongs in the rule.
 
 ---
 
```

## The assembled prompt as sent (17,228 chars)

````markdown
# Role — write the rule from physics, in one shot

You are **translating into code the physics that decides GEMM config
performance on this GPU.**

You are not fixing a particular shape — **express a law.** This rule will be
used on shapes you have never seen, and on other GPUs.

## What you are NOT given — and that is the point

```
no parent rule    you are not editing; you write from scratch
no cases          you are not told "on this shape you were wrong this way"
no scores         no regret, and no indication of which term helped
no measurement table   you do not see this GPU's measurement table
```

This is not about withholding information. **Being able to write a rule
without the table is what makes it portable to a new architecture.** If the
structure only appears after looking at the table, every new GPU needs an
exhaustive sweep — and then there is no reason to have this system.

**When physics is all you can see, you write physics. That is what is being
measured.**

## What you should express

Read the hardware facts and execution model above, and think about **the
paths by which a single kernel gets slower**. For example — this is a
direction of thought, not a list.

```
how much work is done       tiles crossing a shape boundary waste that much
how full the machine is     SMs idle on the last wave
how much memory moves       small tiles re-read A/B many times
whether resources run out   a full smem/register budget cuts resident blocks
whether the pipeline runs   a short mainloop makes fill cost relatively large
whether splitting costs     split-K buys parallelism and sells reduction
```

**Which of these dominates when** is the content of the rule. Branching on a
shape-level value to change a term's weight is how you express that.

## What to put in `changes`

There is no parent, so instead of "what changed" write **one line per term
saying which physics it is**. If you cannot explain a term, drop it.

## ★ Pre-submit checklist — failing here costs a retry

Proposals were discarded three times in a row on these. **Get them all right
in one pass.**

```
[ ] 1. len(w0) == the largest index used + 1, with no gaps
[ ] 2. no shape constant applied to the accumulated score
[ ] 3. at most 4 execution paths
```

The reasoning and the examples are in **"Absolute rules"** above (6, 5, 2).

**The safest shape that satisfies all of them:**

```python
def score(f, p, hw, w):
    s = f.<name> * w[0]
    s = s + f.<name> * w[1]
    s = s + f.<name> * w[2]
    if p.<shape value>:
        s = s + f.<name> * w[3]
    return s
```

**Write the best rule you can.**

⚠️ This variant is the **single-agent control** (D-176 §2). No evolution
loop follows it: whatever this rule is, that is the answer. Do not hold
anything back for a later round.

**Branch where the bottleneck differs.** A memory-bound shape and a
compute-bound shape are not ranked by the same weights, so a split on a
shape-level value (`p.<name>` above) belongs in the rule.

---

## Rule example — ★ built **from the feature list above**

This is a shape example. **Do not submit it as is** — look at why each term
is where it is. Some names are placeholders.

```python
def score(f, p, hw, w):
    # traffic — unbounded, so compress with a log
    s = np.log2(f.<wide-range traffic axis>) * w[0]
    # wave quantisation — already [0,1], leave it
    s = s + f.tail_waste * w[1]
    # spill — binary. Switching it on changes the magnitude, so start large
    s = s + f.has_spill * w[2]
    # resource pressure
    s = s + f.occupancy_deficit * w[3]
    # ★ look at **different physics** per regime (selection, not re-weighting)
    s = s + np.where(p.roofline_ratio < 1,
                     f.<memory-side axis>, f.<compute-side axis>) * w[4]
    return s
```

What this is meant to convey:

```
when to use log compression   for wide-range terms (edge_waste is [0,300])
initial weight of a binary term  large, since switching it on changes magnitude
★ re-weighting vs selection
    if p.x:  s += f.a * w[i]              see the same physics more/less
    np.where(p.x < 1, f.a, f.b) * w[i]    ★ see different physics
one comment per term          the shape of what goes into `changes`
```

**Drop any term you cannot explain.** Each of the five terms above is
explained in one line — that is the bar.

## ★ The split form — each branch has its own weights

When the bottleneck differs, split with `if`. **This differs from
`np.where`** — `np.where` computes both sides, so it is one path, while `if`
splits the path and **each branch carries its own weights.**

```python
def score(f, p, hw, w):
    # common — belongs to every path
    s = np.log2(f.<wide-range traffic axis>) * w[0]
    if p.<regime shape value>:
        # one regime: bandwidth is the bottleneck
        s = s + f.<memory-side axis> * w[1]
        s = s + f.tail_waste * w[2]
    else:
        # the other regime: instructions and occupancy are the bottleneck
        s = s + f.<compute-side axis> * w[3]
        s = s + f.occupancy_deficit * w[4]
    return s
```

```
len(w0) = 5, and the indices run 0..4 with no gaps
★ Split only when the physics differs — not to make the rule look bigger
⚠️ There may be at most 4 execution paths
```


---

## Registered features

These functions already exist, and their definitions are **physical
definitions** — they measure the same thing on another GPU. The brackets are
the **value range**.

## Shape level — access only as `p.<name>`. Scalars, so `if p.<name>:` is allowed
p.alignment_deficit                  [0, 1]  This measures the fractional shortfall of the least-aligned A, B, or C access from the 8-element (16-byte) alignment needed for cp.async-style wide transfers. Poor alignment can require narrower or additional memory transactions and more address/transfer instructions; the deficit is capped at one so it directly represents the unmet fraction. This follows CUDA asynchronous-copy alignment requirements and the supplied alignment semantics.
p.l2_working_set_pressure            [0, 1]  This measures how much the combined A, B, and C working set presses against the GPU's L2 capacity; larger values imply less opportunity for reuse in L2 and more DRAM traffic. The saturating form keeps the quantity bounded while preserving the physical distinction between cache-resident and cache-exceeding shapes. Source: the GPU memory hierarchy/cache-capacity model; the axis is shape- and hardware-only, so it can branch on cache pressure.
p.log_flops                         [0, 80]  log2(2MNK), the absolute size of the problem. It is shape level, so it can be used to branch. It is the axis along which fixed overheads stop mattering. Source: the definition of GEMM's flop count, 2MNK (NVIDIA Matrix Multiplication Background User Guide)
p.log_min_dim                       [0, 30]  log2 of the shortest of M, N, K. It is shape level, so it can be used to branch. A dimension shorter than the tile that covers it throws work away whatever else is chosen. Source: the GEMM problem statement — M, N, K are the inputs (NVIDIA Matrix Multiplication Background User Guide)
p.output_aspect_imbalance            [0, 1]  This measures output-matrix rectangularity as the normalized difference between M and N. Greater imbalance means less symmetric row/column parallelism and operand reuse, which changes the useful tile geometry and scheduling regime even when total FLOPs and arithmetic intensity are similar. It is computed only from the shape, so a rule can branch on square-like versus highly rectangular GEMMs; the ratio is bounded by 1.
p.output_work_per_sm         [1e-09, 1e+18]  This measures the output surface available per SM, derived as M*N independent output elements divided by the number of SMs. Very small values indicate insufficient spatial work to keep all SMs occupied, while large values describe a different regime where per-SM saturation and kernel efficiency dominate; it is distinct from total FLOPs because K does not enter. The shape/hardware-only formula is branchable and follows the GPU parallel-work model.
p.roofline_ratio                 [0, 10000]  Arithmetic intensity / ridge point. Below 1 it is memory-bound. It is shape level, so it can be used to branch, and it handles the boundary more smoothly than a binary verdict. Source: Williams, Waterman, Patterson (2009), "Roofline"

## Config level — access only as `f.<name>`. **Arrays** (no `if`, raises ValueError)
f.edge_waste                       [0, 300]  The multiple of work thrown away where a tile crosses the shape boundary, minus 1. With a 128-row tile on M=1 the value is 127. The range is wide (hundreds), so give it a small weight or compress it with a log. Source: CUTLASS documentation, predication
f.has_spill                          [0, 1]  Registers overflow into local memory (= DRAM). If a register access is one cycle, local is hundreds, and it happens on every iteration inside the mainloop. Source: CUDA C++ Best Practices Guide, "Register Pressure"
f.instruction_density            [0, 1e+09]  This measures static instruction overhead relative to the useful arithmetic performed by a CTA's output tile over the full K reduction. Higher instruction density leaves fewer issue slots for tensor arithmetic and increases front-end/control overhead; it is distinct from memory traffic and occupancy because it measures work per instruction rather than bytes or resident resources. The normalization follows the GEMM FLOP count, 2MNK, applied to one tile.
f.k_tail_waste                       [0, 1]  This measures reduction-dimension predication: a mainloop must issue work for the final tile_k-sized K slice even when K ends partway through it. The resulting padded tensor operations consume compute and operand bandwidth without contributing to the GEMM result; unlike edge_waste, this isolates the K boundary rather than M/N output boundaries. It follows the tiled GEMM/predication model used in CUDA GEMM implementations.
f.mainloop_iteration_count       [1, 1e+18]  This measures the absolute number of reduction mainloop iterations, including the final partial slice. Each iteration incurs loop-control and pipeline synchronization/barrier activity, so the count expresses exposed per-iteration latency; unlike pipeline_fill_drain_fraction it is not normalized away for long reductions. The ceiling follows tiled GEMM execution, where a partial final K tile still requires an iteration.
f.occupancy_deficit                  [0, 1]  The fraction of thread slots per SM left unfilled. There is less room to hide memory latency. It matters less when compute-bound. Source: CUDA Occupancy Calculator
f.operand_tile_traffic_ratio     [1, 1e+06]  This measures global-memory operand traffic caused by loading an A and B tile for every output tile and every split-K slice, normalized by the compulsory A+B bytes. Larger values mean more replicated loads and less inter-CTA operand reuse, increasing bandwidth demand; it is a memory-traffic axis distinct from shape-only L2 pressure and the separate partial-reduction traffic feature. The formula is the tiled GEMM data-movement model.
f.parallel_reduction_compute_fraction         [0, 1]  Parallel split-K requires an additional reduction tree: each output element combines `split_k` partial accumulators, adding `split_k - 1` accumulator operations beyond the ordinary GEMM accumulation. Dividing by the GEMM's `2K` FLOPs per output expresses the extra compute fraction; unlike parallel reduction traffic, this measures arithmetic work rather than bytes. The derivation follows the standard parallel split-K reduction model.
f.parallel_reduction_traffic_fraction         [0, 1]  Parallel split-K materializes partial accumulators in global memory and subsequently reads them for the reduction; the factor of two represents those write and read streams. Normalizing per output element by the ordinary output-store traffic gives the fraction of output-equivalent traffic attributable to partial reduction, so it is bounded and remains distinct from shape-only cache pressure. This follows the standard parallel-reduction memory-traffic model; serial split-K keeps partials in the accumulator and contributes zero here.
f.pipeline_fill_drain_fraction         [0, 1]  This measures pipeline fill and drain overhead relative to the number of K mainloop iterations. More stages require more prologue/epilogue steps, and that fixed latency is amortized only when K supplies a long steady-state loop; the capped fraction reaches 1 when no steady-state interval remains. It is a pipelining axis distinct from occupancy and memory/cache pressure.
f.register_file_pressure             [0, 1]  A CTA's register allocation is approximately its registers per thread multiplied by its resident threads. A larger fraction of the SM register file reduces the register capacity available for concurrent CTAs and can limit latency hiding, independently of thread-slot occupancy and shared-memory allocation. This follows the CUDA occupancy/resource-allocation model.
f.resident_block_scarcity            [0, 1]  This measures the scarcity of independently resident CTAs on each SM: the inverse residency count is larger when resource limits allow fewer concurrent blocks, reducing the scheduler's pool for latency hiding. It is distinct from thread-slot, register-file, and shared-memory pressure because it captures the resulting CTA-level residency limit rather than one resource fraction. The derivation follows the CUDA occupancy/resource-allocation model.
f.serial_reduction_dependency         [0, 1]  This measures the fraction of available K-loop iterations affected by serial split-K boundaries. Each additional serial slice must wait for the prior partial accumulator, reducing independent scheduling opportunities; unlike parallel_reduction_traffic_fraction, it measures dependency/parallelism loss rather than DRAM traffic. The formula is bounded by one because it is a normalized dependency fraction.
f.shared_memory_pressure             [0, 1]  This measures the CTA's shared-memory allocation as a fraction of the hardware per-block budget. A larger fraction leaves less room for additional resident CTAs and can constrain shared-memory buffering, which is a resource-pressure axis distinct from thread-slot occupancy and register spilling.
f.spill_traffic_ratio            [0, 1e+18]  This estimates the bytes written to and read from local memory when register spills occur, including each thread and K-loop iteration, then normalizes them by the compulsory A, B, and C traffic. Local memory is backed by the cache/DRAM hierarchy and is far slower than register access, so the traffic ratio measures spill severity rather than merely spill presence; it is distinct from register-file pressure and the binary spill indicator. The read/write factor follows the usual memory-traffic accounting for spilled values.
f.tail_waste                         [0, 1]  The fraction of SM slots idle in the last wave. The loss is roughly 1/(1-x)x — 2x at 0.5, 5x at 0.8. Used linearly, that magnitude does not come out. Source: CUDA C++ Best Practices Guide, "Thread and Block Heuristics"
f.thread_work_granularity_deficit         [0, 1]  A CTA pays per-thread scheduling, coordination, and instruction-issue overhead while producing its tile's output elements. This feature measures that fixed thread overhead relative to the tile's useful output surface; larger values mean less output work amortizes the CTA's thread-level overhead. It is distinct from edge waste because it uses the configured tile's internal work granularity rather than shape-boundary overhang, and distinct from occupancy because it does not measure resident CTAs or SM resource limits.
f.tile_aspect_mismatch              [0, 20]  This measures the multiplicative mismatch between the configured CTA's row/column geometry and the GEMM output geometry. A mismatch can make partitioning and A/B operand reuse less balanced across the two output dimensions; unlike edge waste, it remains nonzero even when tiles divide the shape exactly. The absolute logarithm is zero for matching aspect ratios and grows symmetrically for either kind of mismatch.
f.tile_roofline_deficit              [0, 1]  This measures the arithmetic-intensity deficit of one CTA's A/B tile loads relative to the hardware roofline. Low tile intensity means each fetched operand supports little tensor arithmetic, so bandwidth pressure can dominate even when the whole GEMM's shape-level intensity is high; the deficit is capped at one and is distinct from total operand replication traffic.
f.wave_count_deficit                 [0, 1]  This measures the inverse number of CTA waves needed to dispatch the grid, capturing fixed launch and grid-scheduling overhead that is not represented by the fractional idle loss of the final wave. A one-wave kernel has value 1, while many waves approach 0 as startup cost is amortized; it is derived from the GPU CTA scheduling model and uses the available SM/CTA slots.

★ Swapping the prefix is rejected immediately. Using a name that is not in the `p.` list as `p.`, or the reverse.

---

## Table aggregates

**None.** That is condition A — write from physics alone.
````
