# HeteroWasm

HeteroWasm is a conservative compiler from ordinary WebAssembly to WebGPU.

It takes a WebAssembly program and finds the data-parallel loops that can be shown safe to run together. Only those regions become a WebGPU compute shader. The original program remains the CPU implementation: a region stays there when it cannot be shown safe, and it stays there when the GPU would be slower.

The shader is generated from the program. It is not a second implementation written by hand. Where a region never leaves the CPU, the original WebAssembly is the program that runs.

## How it compiles

1. **Discovery.** Natural loops, induction variables, affine addresses, loop bounds. A loop that cannot be reduced to that form is out.
2. **Proof.** Four obligations — control, memory, dependence, effects. A loop is offloaded only when every one is proven, or when a runtime guard can cover it.
3. **Cost.** Fewer than 32 multiplies in the loop body, or fewer than 65536 dynamic trips, and the loop stays on the CPU. Both numbers come from measurements on this machine, not from a guess.
4. **Emission.** Each qualified loop becomes a WGSL compute kernel plus a manifest; the module is rewritten so the loop turns into one `__hw_dispatch` call. Everything else keeps the original code.

The analysis of a function runs **once** per compile: the same decision drives kernel emission and the module rewrite, so the two do not re-analyse the program. Fusion (two loops merged into one kernel) is planned only inside functions that can contain a fusable pair — a loop that writes an intermediate region and a second loop that reads it, both in the same function and both lowerable.

## Where the work stands

The compiler is still under development. The GPU path measured so far is Metal, on an Apple M2.

Degree-64 integer polynomial, 65536 elements, `--repeat 5`, memory restored before each timed call:

| | time |
| --- | --- |
| CPU | 3.1786 ms |
| GPU | 0.5832 ms |
| | **5.45× faster**, memory matches |

An earlier run of the same loop was 3.69×. The ratio moves. The memories matched both times. If they do not match, no ratio is reported.

Same machine, same protocol, `n = 65536`:

| Loop | CPU | GPU | What the compiler does |
| --- | --- | --- | --- |
| 16-step Horner | 0.3300 ms | 0.4753 ms | stays on the CPU, 1.44× slower on the GPU |
| 32-step Horner | 0.9389 ms | 0.5267 ms | offloaded, 1.78× faster |
| degree-64 polynomial | 3.1786 ms | 0.5832 ms | offloaded, 5.45× faster |

A later run of the 32-step Horner was 1.37× faster. Under 32 multiplies in the body, or under 65536 trips, the loop stays on the CPU. At `n = 32768` the 32-step Horner was still slower.

The loops behind those numbers are integer load, store, add, subtract, and multiply, in one loop. These stay on the CPU in this snapshot:

- floating-point arithmetic
- division, remainder, shifts, bitwise operations
- a reduction such as `s += in[i]`
- nested loops, indirect indexes, unaligned access
- fewer than 32 multiplies, or fewer than 65536 trips

A short loop is cheaper on the CPU.

### Where the gate sits

Two gates decide whether a loop reaches the GPU: one at compile time on the body, one at
run time on the trip count.

Compile gate, this build, `--repeat 5`, output at byte 0, input at byte 524288, 1 MiB of
linear memory:

| body | multiplies | kernel emitted |
| --- | --- | --- |
| Horner 16 | 16 | no |
| Horner 24 | 24 | no |
| Horner 32 | 32 | yes |
| Horner 40 | 40 | yes |

Runtime gate, same machine, same kernel present:

| body | n | dispatched | CPU | GPU | verdict |
| --- | --- | --- | --- | --- | --- |
| Horner 32 | 32768 | no | 0.55 ms | — | no dispatch; the CPU path runs |
| Horner 32 | 65536 | yes | 0.91 ms | 0.54 ms | 1.68× faster |
| Horner 40 | 32768 | no | 0.64 ms | — | no dispatch; the CPU path runs |
| Horner 40 | 65536 | yes | 1.27 ms | 0.54 ms | 2.38× faster |

At `n = 131072` this buffer is too small for the access range: the run-time bounds guard
keeps the CPU path because the Wasm would trap while the GPU would not.

The four Horner bodies are `corpus/synthetic/intensity/horner{16,24,32,40}.wat`.

## Compiling

Rust 1.98 or newer.

```bash
cargo build -p heterowasm-cli --release
heterowasm compile corpus/real/draco_decoder.wasm --output out
```

`compile` writes one directory:

| file | contents |
| --- | --- |
| `original.wasm` | the input module — the CPU implementation and the correctness reference |
| `<kernel>.wgsl` + `<kernel>.json` | one compute shader and its manifest (fields, results, dispatch) |
| `kernels.json` | the kernels in the order the rewritten module calls them |
| `rewritten.wasm` | the module with the offloaded loops replaced by `__hw_dispatch` calls |

Real inputs on the same machine, release build, median of six runs:

| program | size | compile |
| --- | --- | --- |
| Draco decoder | 286 KB | ≈0.5 s |
| Basis transcoder | 490 KB | ≈0.35 s |

On the Draco build the rewritten program dispatches nothing: one loop clears the gate, and the decode path does not enter it. [docs/BOUNDARIES.md](docs/BOUNDARIES.md) has the per-stage counts and how to read the refusal lines — `compile` prints one `skip` line per refusal, and `analyze --trace-jsonl` writes one event per obligation.

`compile` ends with a one-line breakdown of why loops stayed on the CPU: how many contain
calls or other side-effecting operators, how many failed legality split by obligation
(control, memory, dependence, effects), how many fell below the cost gate, and how many
could not be lowered. `--require-offload` turns "no kernel was produced" into a non-zero
exit, for build scripts that require the GPU path.

To inspect decisions without writing artifacts:

```bash
heterowasm analyze corpus/real/draco_decoder.wasm --json
heterowasm analyze corpus/real/draco_decoder.wasm --trace-jsonl trace.jsonl
```

## Running and measuring an artifact

```bash
heterowasm run out --entry main --arg 0 --arg 524288 --arg 65536
heterowasm bench out --entry main --arg 0 --arg 524288 --arg 65536 --repeat 5
```

The three arguments are the output byte offset, the input byte offset, and the element count. Those two buffers do not overlap. `--repeat 5` is the number behind the table above. `bench` runs the CPU and GPU paths, compares the memories first, and only then reports a ratio — a result that is wrong is never fast.

The buffers are either exactly the same range, in place, or fully apart. A partial overlap is rejected: parallel iterations would see the original memory, and the Wasm loop would see values written by earlier iterations.

## Testing

```bash
cargo test --workspace
```

The suite currently reports 168 passing tests. Tests that read the large compiled corpus artifacts skip when those artifacts are absent, so a clean clone is green.

## License

MIT. See [LICENSE](LICENSE).
