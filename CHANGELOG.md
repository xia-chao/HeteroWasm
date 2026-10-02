# Changelog

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
