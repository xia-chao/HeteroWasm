# Changelog

## Unreleased

- `compile` prints a breakdown of why loops stayed on the CPU: side-effecting operators,
  legality split by obligation, the cost gate, lowering, and loops without an exit.
- `compile --require-offload` exits non-zero when no loop was emitted.
- `analyze --json` gains `summary.cpu_reasons`, counting CPU loops by the obligation that
  decided them (`proven_illegal` and `unknown`, per obligation kind).
- `analyze` prints the same breakdown as "why the rest stays on the CPU".
- Added Horner bodies with 24 and 40 multiplies; README now has a measured table for both
  gates (32 multiplies at compile time, 65536 trips at run time).

## 0.1.0

Source release. No published binary. The GPU path is measured on Apple M2 / Metal only.

A loop is rewritten onto the GPU only when both of these hold:

- the shader contains at least 32 multiplies
- the trip count is at least 65536

Otherwise the original Wasm runs on the CPU. Correctness is checked before a speedup is
reported. If the memories differ, `bench` does not print a ratio.

Measured with `--repeat 5`, memory restored before each timed call, output at byte 0,
input at byte 524288, `n = 65536`:

| Workload | CPU | GPU | Verdict |
| --- | --- | --- | --- |
| 16-step Horner | 0.3300 ms | 0.4753 ms | 1.44× slower, not offloaded |
| 32-step Horner | 0.9389 ms | 0.5267 ms | 1.78× faster |
| degree-64 polynomial | 3.1786 ms | 0.5832 ms | 5.45× faster |

Ratios move between runs. Those three verdicts did not.

Not in this release: floating-point operators, reductions, gathers, a Vulkan or DX12
measurement, and a real decoder call that is faster than the CPU.
