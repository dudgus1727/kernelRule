<!-- ★ Fixed areas (§30.18). Letting the LLM pick them on every run means a
     fourth list could differ, and then **that run alone has a different
     condition** — silently. Same logic as D-45 (model constant) and D-47
     (seed not passed): do not leave controllable things uncontrolled.

     These seven cover all three lists the LLM produced (comparison in
     §30.18). Only an explicit `--recategorize` re-derives them with an LLM. -->

## Physical areas — spread your work evenly over these seven

```
compute throughput   | how well the compute resources are used
memory traffic       | how much data moves
compute/traffic ratio| which of the two is the bottleneck. `hw.ridge_point`
                       is that boundary
resource pressure    | how close you are to the per-SM resource limits
pipelining           | how well latency is hidden
reduction            | the cost of combining partial sums
alignment & edges    | handling of data boundaries and shape boundaries
```

**An area is only "where to measure", not what to measure.** Which physical
quantity you build inside it is your decision. Several per area is fine, and
skipping one is fine if there is nothing to express there.
