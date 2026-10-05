<!-- D-193 — the size guidance for condition F4 (`--size-guidance
     role/_size_time.md`). Same wording as _size_loop.md except the first
     paragraph and the last: with the time features the main term is one
     time estimate, and the memory/compute split is inside it, not a branch. -->
That is the smallest honest shape — with one change when the feature list
holds time features: the first lines are **one estimate of the kernel's
time** (the example below), and every term after it is a correction to that
estimate.

{size_bar}

**A branch is for a correction whose physics differs between shapes.** The
time estimate already tells a memory-bound config from a compute-bound one —
inside every shape, config by config — so that split is not a branch on a
shape-level value. Split only when a correction acts differently on, say,
skinny and square shapes, and name that physics.
