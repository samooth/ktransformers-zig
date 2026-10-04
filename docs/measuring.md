# Measuring — what a number in this repo is allowed to mean

Read this before you quote a ratio, before you trust one you find in a doc, and
before you conclude that a change did nothing. Most of the rules here were
learned the expensive way, by three separate sessions converging on the same
answer from different directions.

## The one rule

**Publish a ratio only with a byte-identical control measured in the same run.**
If the control cannot resolve the difference, print "below the noise floor" and
stop. A number that invites a decision it cannot support is worse than no number.

## The noise floor is load-dependent, not a constant

**This section previously said "the noise floor is 14-23%" and "nothing under
~25% is measurable". Both were wrong, and wrong in the exact way this document
exists to prevent: a measurement taken under load was generalised into a law, by
me, about noise.**

The 14-23% figures are real but they were measured at **load 17-80 on 16
cores**. Re-measured with a control arm on each side
(`bench/cpp_vs_zig/measure.zig`, three passes each), the floor tracks load:

| load average (16 cores) | measured floor | what resolves |
|---|---|---|
| **2 - 4** | **1.02x - 1.14x** | most rows, including `m>=8` GEMM |
| 17 - 27 | ~1.17x | fewer |
| ~80 | ~1.25x | little |

So the corrected rule:

- **On a quiet box the floor is ~1.02-1.14x, not 1.25x.** A 15-20% difference is
  measurable there, and the five-format IQ and the `m>=8` GEMM rows are all in
  that band.
- **The floor must be measured in the same pass as the number it judges**, with a
  same-code control arm on each side. Never quote a floor from an earlier
  session, and never quote a ratio without the floor beside it.
- **State the load average with every figure.** It is not context, it is the
  variable that sets whether the figure means anything.
- **Prefer a quiet box to more iterations.** Going from load 80 to load 3 moved
  the floor from 1.25x to 1.04x; going from 3 to 9 iterations on a busy box does
  not buy the same thing, because the spread being measured is the host's, and
  more samples of a noisy host estimate the same noisy host.

The general form of the mistake is worth stating, since I made it here: a floor
measured under one condition and quoted under another is not a conservative
estimate, it is a fabricated one. It reads as rigor and delivers the opposite.

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

## Adding an enum variant: iterate the enum, never a hand-written list

This is a shape failure, not a review failure, and it is worth separating those
two things because "be careful" does not fix it.

What happened, concretely: `bench/cpp_vs_zig/measure.zig` grew a **fourth arm**
(a same-code control on the Zig side, so the floor could bound both sides of a
quotient). Four loops iterate the arms. Two of the four were updated to name all
four; two still listed three. The two that were missed are exactly the ones that
computed the statistics, so `ctl`'s `lo`/`hi`/`best` stayed `undefined` — and
**the Zig-side control contributed nothing to any floor, silently.** Half the
measurement was missing from every number the tool printed.

Three things made it survive:

- `undefined` in ReleaseFast is usually benign. The floors came out plausible
  and nobody could tell.
- A `TODO`-free, compiling, plausible-output state is not a state any review
  flags. It looks finished.
- It only became visible as an `inf` floor under `--self-test`, and the
  Debug-mode allocator was reporting six *arena* leaks at the same moment, which
  is what actually led me to look.

The fix is one line, and it is a shape fix:

```zig
// WRONG: adding a variant means finding every tuple that lists them.
for ([_]Arm{ .cpp, .zig, .ctl }) |arm| { ... }

// RIGHT: adding a variant cannot break it.
for (std.enums.values(Arm)) |arm| { ... }
```

The same applies to every dispatch table, every `inline for` over a
hand-maintained tuple of cases, every "and the third call site" edit. I had four
call sites and edited three; that is not carelessness, it is the shape inviting
it. Anything that enumerates cases should enumerate the enum.

Two habits that follow from the same incident:

- **Initialise aggregates to a defined value, not `undefined`,** when they are
  filled by a loop. `undefined` turns a missed iteration into a wrong-but-finite
  number that survives review. `@splat(0)` turns it into something a self-test
  catches.
- **A self-test is only an asset if it can fail.** The whole reason this was
  caught is that `--self-test` asserts something checkable — with both arms the
  same binary the true ratio is 1.0, so every row must read UNMEASURABLE. A
  check that cannot fail is a comment, and the previous two implementations of
  the floor passed a casual read while failing it.

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
