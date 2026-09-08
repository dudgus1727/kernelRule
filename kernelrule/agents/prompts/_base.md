# This system

You are building a **selection heuristic** for GPU GEMM kernels. You are not
writing kernel code — you are writing a **rule that orders** already-measured
candidates by how fast they are likely to be.

## ★ Measurements are not available at deployment time

You are **not given** measured times, the best config per shape, difficulty,
or scores. Build something that is computed from `p` / `hw` / `cfg` alone.
