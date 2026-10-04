# Measuring — what a number in this repo is allowed to mean

Read this before you quote a ratio, before you trust one you find in a doc, and
before you conclude that a change did nothing. Most of the rules here were
learned the expensive way, by three separate sessions converging on the same
answer from different directions.

## The one rule

**Publish a ratio only with a byte-identical control measured in the same run.**
If the control cannot resolve the difference, print "below the noise floor" and
stop. A number that invites a decision it cannot support is worse than no number.

## The noise floor is 14–23%

A copy of a binary that was byte-identical to the baseline differed from it by
**14% to 23%** on the same shapes, interleaved, pinned, min-of-N. This is the
single most important fact on this page:

- **Nothing under ~25% is measurable.**
- A reported "1.2x" on this host is indistinguishable from a 23% regression.
- If a change is real, it should be worth far more than 25% — the kernel work
  that landed landed at 3x–165x, not 1.2x.

## Absolute time moves by up to 3x with load

The host is shared. It has been observed at load 54/19/45 on 16 cores. The same
case measured 10x, 14x and 19x on three consecutive runs under varying load.
Therefore:

- **Never compare against a remembered number.**
- **Always state the load average** with a figure.
- A ratio measured under load is a **lower bound**. Most published ratios in this
  repo are floors for exactly this reason, and that is fine — but say so rather
  than implying they are final.
- Pin with `taskset -c <cpu>` so two runs are comparable.

## Microbenchmarks of a kernel fragment invert here

A fragment that accumulated into a single `f32` reported vector ops **14% slower**
than scalar, while the same code inside the real GEMM measured **165x faster**.
Both numbers came from the same session on the same day. The causes:

1. **Latency, not throughput.** A single accumulator measures the dependency
   chain; a real kernel measures how much work hides behind it.
2. **Load 13–51** during the fragment measurement.

So: measure at the GEMM level, with both paths in one binary, back to back.
A fragment number is a hypothesis, never a result.

## `perf` is not available

`perf_event_paranoid=4`. Any workflow that bisects with `perf` is measuring the
wrong thing. Use `objdump -d` for instruction counts and A/B timing for
behaviour.

## The only trustworthy recipe

Both implementations in **one binary**, interleaved, pinned, min-of-N:

```
taskset -c 9 <binary>          # both paths, same process, alternate them
```

Report, together, every time:

| field | why |
|---|---|
| ratio | the number |
| **N** (runs, min-of) | a single sample is not a measurement |
| **control spread** | byte-identical baseline vs itself — the noise floor for this shape |
| **load average** | whether the host was busy |
| absolute ms or GB/s | ratios hide which side moved |

If the ratio is inside the control spread, the honest report is "unmeasurable,
below the floor" — not a number.

Two further rules:

- **Never measure in Debug.** Benchmark with `-Doptimize=ReleaseFast`.
- **Prefer absolute throughput (GB/s) to a speedup ratio** when the baseline is
  questionable. A 165x speedup against a 55-cycles-per-weight baseline is a
  statement about the baseline, not about the kernel. `1.26 GB/s at m=1` places a
  format correctly next to its siblings; `69x` places it nowhere.

## Benchmarks that do not exercise the kernel

A checksum is not correctness: a flat sum is permutation-invariant, so it cannot
prove that outputs landed in the right columns. For correctness, compare
**element-wise against the scalar oracle**, and mutation-test the test — drop a
term, confirm it fails, restore it. That is what gives a test teeth.

Shape coverage matters too: if a kernel vectorises over **k** (not n), a small
`n` is harmless and the vector body still runs. Check which axis is vectorised
before calling a shape a gap.

## Numerical acceptance is a budget, not a zero

Pinning a test to "bit-identical" can fail on legitimate data. Two examples from
this repo:

- bf16 has exact rounding **midpoints**; two correctly-rounded implementations
  can straddle one and differ in the last bit. The activation test therefore
  pins a **1-in-100000** budget, not zero — the benchmark's own inputs happened
  to give 0 of 262144, but a different distribution gives 2 of 262144.
- Where a test does pin exact equality, it must be equality against the
  **documented semantics** (e.g. "the exponent nibble comes out of `aux1`'s top
  4 bits"), not against a reimplementation of the format that happens to agree.

## Where these numbers came from

| finding | session |
|---|---|
| 14–23% noise floor against a byte-identical control; register-pressure diagnosis disproved | SwiGLU / P1.3 |
| fragment microbenchmark inverted by 165x; latency-vs-throughput; `perf` locked down | IQ2_XXS |
| accuracy test missing entirely on the vector path every activation runs; 1-in-100000 budget | SwiGLU / P1.3 |
| SIMD numbers in Debug are meaningless; pin and interleave | earlier |
