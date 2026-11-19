# Boundaries on a real binary

The README table uses small synthetic loops. This page records what the compiler does on a real
workload: a 286 KB Draco decoder binary. It is here so that the size of the GPU path is visible
before someone points the tool at their own program.

## How to read the output

Every loop that stays on the CPU leaves one line on stderr:

```console
$ heterowasm compile draco_decoder.wasm --output out 2>&1 | grep '^skip'
skip : unsupported_field_affine
skip : no_extent
skip : unaligned
...
```

The line names the last refusal on the path, not every reason the loop failed. For the full set of
per-obligation verdicts, ask for the event stream:

```console
$ heterowasm analyze draco_decoder.wasm --trace-jsonl trace.jsonl
```

The trace carries one event per obligation (control, memory, dependence, effects) with its verdict
and reason, plus the loop-level disposition (`gpu`, `gpu_after_guard`, `cpu`).

`compile` also ends with a one-line breakdown. It reports the **first** refusal it reaches, so loops
that contain calls or other side-effecting operators are counted there — before any obligation is
judged — and the rest are split by the obligation that decided them. `analyze` runs the full chain
and reports the decisive obligation instead, which is why the two breakdowns do not have to match
number for number.

## The numbers

Numbers below are from one pass over the decoder: 962 natural loops.

| stage | count |
| --- | --- |
| legal, no guard needed (`gpu`) | 7 |
| legal with an alias or bounds guard (`gpu_after_guard`) | 124 |
| emitted to a kernel once the cost gate is taken out of the picture | 64 |
| emitted with the real cost gate in place | 1 |

The real decode path dispatches nothing: the rewritten program is bit-identical to the original and
runs on the CPU. The single kernel (`10-23`) is the only loop in the binary with enough multiplies
per iteration to clear the gate.

The other 63 lower to kernels but are copies and fills: 4 to 19 multiplies per iteration. They stay
on the CPU because the gate is a measurement, not a guess. Widening the multiply gate to 8 moved
four loops of the decode onto the GPU and made the program slower: 172 ms against 102 ms.

## Why the gap is density, not reach

An earlier line of work widened the analysis to accept loops whose uniform fields are computed
inside the loop (a common `-O0` shape). It converted 41 `unsupported_field_affine` refusals into
lowerable kernels — and none of them produced a kernel that passed the cost gate: their bodies have
0 to 17 multiplies. Opening the analysis boundary does not create arithmetic.

That is the current shape of the project: the boundary that keeps loops on the CPU is usually a
proof obligation on a loop that would have been too small anyway. Both matter, and they are
measured separately.

## What would move it

- A loop with at least 32 multiplies per iteration and at least 65536 dynamic trips, with affine
  addresses and no loop-carried dependence, is the shape that qualifies today.
- More work per dispatch is the only way to bring smaller loops across without lying about the
  measurement: merging independent loops into one dispatch, or fusing a producer into its
  consumer. Both are measured, not assumed, and neither is in this build.
- Lowering the gate is not a fix. It publishes a program that is slower than the one it replaces.
