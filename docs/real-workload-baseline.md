# Real-workload baseline

This document records what the compiler currently does on a real Draco decode,
measured with the `draco_decode` example. It exists because kernel benchmarks are
not enough: a kernel can be fast and still never run on a real workload.

## Samples

| Sample | Bytes | sha256 | Decode result |
| --- | --- | --- | --- |
| `bunny.drc` (draco3d@1.5.7) | 96018 | `0cfb7a00…a48cc` | 34834 points / 69451 faces |
| `cube.drc` (google/draco testdata) | 199 | `f90c7377…f7caa` | 24 points / 12 faces |

Both are decoded through the same call sequence on two hosts: raw wasmtime with the
original module, and `Runtime` with the rewritten module. Geometry data and the whole
linear memory are compared word for word.

## Result

With the default cost gates, `draco_decoder.wasm` emits a single kernel (the 64-byte
`memcpy` block loop) and that loop never runs for these samples: **zero GPU
dispatches**, output bit-identical. This is the correct behaviour, not a dropped
call; the emission surface simply does not cover the hot decode loops yet.

Compile diagnostics for the same module: 958 loops skipped — 397 with side effects
(calls, bulk memory, global/table writes), 431 by legality (251 memory, 180
dependence), 79 below the cost gate, 51 not lowerable.

Even lifting the cost gates only produces 64 kernels, and a real decode runs nine
dispatches (selectors 22, 29, 29, then 1 six times) before a later kernel is
correctly refused by the runtime bounds contract (negative-offset headroom). The
total transferred data is 1.68 MB, and no dispatch consumes another one's output,
so a device-resident execution model would have nothing to save today.

## Host overhead

A single bunny decode calls exported functions 382,322 times. The `Runtime` host used
to be 1.6-1.8x slower than raw wasmtime on that decode even with zero dispatches.
Per-call probes attributed the difference to trace work that was built but never
recorded: the `run` stage scope (~38 ms) and the `call finished` info event
(~28 ms) together accounted for all of it. `Runtime::run` now asks
`Trace::enabled(level)` before building either, so a filtered trace costs nothing.
After the fix the two hosts are within measurement noise of each other
(original 95-108 ms, rewritten 100-107 ms over five runs), and the rewritten path
is still bit-identical.

## What this means

Real-workload speedups cannot be claimed from kernel benchmarks alone: with the
current emission surface there is no GPU work on the real path, and the remaining
blockers are the 397 side-effect loops (98.5% of them contain calls) and the 431
legality rejections. Those are interprocedural and aliasing problems, not another
runtime feature.
