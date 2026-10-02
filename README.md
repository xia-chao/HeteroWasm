# HeteroWasm

HeteroWasm is a conservative compiler from ordinary WebAssembly to WebGPU.

It takes a WebAssembly program and finds the data-parallel loops that can be shown safe to run together. Only those regions become a WebGPU compute shader. The original program remains the CPU implementation: a region stays there when it cannot be shown safe, and it stays there when the GPU would be slower.

The shader is generated from the program. It is not a second implementation written by hand. Where a region never leaves the CPU, the original WebAssembly is the program that runs.

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

Rust 1.98 or newer.

```bash
cargo build -p heterowasm-cli --release

heterowasm compile corpus/synthetic/intensity/poly64.wat --output out
heterowasm bench out --entry main --arg 0 --arg 524288 --arg 65536 --repeat 5
```

The three arguments are the output byte offset, the input byte offset, and the element count. Those two buffers do not overlap. `--repeat 5` is the number behind the table. The loop writes `out[i]` from a polynomial in `in[i]`. The buffers are either exactly the same range, in place, or fully apart. A partial overlap is rejected: parallel iterations would see the original memory, and the Wasm loop would see values written by earlier iterations.

## License

MIT. See [LICENSE](LICENSE).
