# HeteroWasm

**The slow loop stays in Wasm. The heavy one runs on the GPU.**

You already have a WebAssembly module. HeteroWasm compiles it, runs the loops that pay
for a dispatch, and leaves the rest on the CPU. You do not write a second program in
WGSL, and you do not connect wasmtime to wgpu yourself.

On an Apple M2, a degree-64 integer polynomial of 65536 elements:

| | time |
| --- | --- |
| CPU | 3.1786 ms |
| GPU | 0.5832 ms |
| | **5.45× faster**, memory matches |

An earlier run of the same loop was 3.69×. The ratio moves. The result stayed faster,
and the memories matched both times. If they do not match, `bench` prints no ratio.

The measured GPU is Metal.

---

## What you get

One command compiles. One command runs. One command tells you whether the GPU was
worth it.

```text
out/
  original.wasm     the module you passed in
  rewritten.wasm    the same program, with the heavy loop turned into a host call
  poly64-1-3.wgsl   the compute kernel that call runs
  kernels.json      which call uses which kernel
```

`run` needs nothing else installed beside the `heterowasm` binary. `bench` runs the
original module and the rewritten one, restores memory before every timed call, and
only then prints a verdict.

A loop that would be slower stays in `original.wasm`. A loop that cannot be shown safe
stays there too. You can always run that file and ignore the GPU path.

---

## Reproduce the number

Rust 1.98 or newer.

```bash
cargo build -p heterowasm-cli --release

heterowasm compile corpus/synthetic/intensity/poly64.wat --output out
heterowasm bench out --entry main --arg 0 --arg 524288 --arg 65536 --repeat 5
```

```text
correctness   CPU and GPU memory **match word-for-word** ✓
CPU           3.1786 ms/run
GPU           0.5832 ms/run
verdict       GPU **5.45× faster**
```

The three arguments are the output byte offset, the input byte offset, and the element
count. Those two buffers do not overlap. `--repeat 5` is the number to trust. A single
run is not.

The source of that loop is `corpus/synthetic/intensity/poly64.wat`: one function, one
loop, `out[i]` written from a polynomial in `in[i]`.

---

## Point it at your module

```bash
heterowasm compile your-module.wasm --output out
heterowasm bench out --entry your_export --arg <out> --arg <in> --arg <n> --repeat 5
```

WAT and wasm both compile. The loop HeteroWasm can take off the CPU looks like this:

```text
for i in 0..n {
    out[i] = a long integer polynomial in in[i]
}
```

`n` is an argument. The buffers are either exactly the same range (in place) or fully
apart. A partial overlap is rejected. The message starts with `spec §15 rejects GPU`.
Parallel iterations would see the original memory. The Wasm loop would see values
written by earlier iterations. Those are not the same answer.

---

## When it is faster

A dispatch costs a few tenths of a millisecond before the GPU does useful work. The
CPU time has to clear that. Same machine, same protocol, `n = 65536`:

| Loop | CPU | GPU | What happens |
| --- | --- | --- | --- |
| 16-step Horner | 0.3300 ms | 0.4753 ms | stays on the CPU, 1.44× slower on the GPU |
| 32-step Horner | 0.9389 ms | 0.5267 ms | offloaded, 1.78× faster |
| degree-64 polynomial | 3.1786 ms | 0.5832 ms | offloaded, 5.45× faster |

A later run of the 32-step Horner was 1.37× faster. Still a win. The tool does not
lower the bar to chase one ratio.

Under 32 multiplies in the body, the loop stays on the CPU even at this length. Under
65536 trips, it stays on the CPU even for the 32-step Horner: at `n = 32768` that
loop was still slower. Copies, stencils, and short convolutions are on that side of
the table. `bench` is how you place a module that is not listed here.

---

## What stays on the CPU

The fast path above is integer load, store, add, subtract, and multiply, in one loop.

These stay on the CPU:

- floating-point arithmetic
- division, remainder, shifts, bitwise operations
- a reduction such as `s += in[i]`
- nested loops, indirect indexes, unaligned access
- fewer than 32 multiplies, or fewer than 65536 trips

That is a limit of this release, not a hint that a short loop was forgotten. A short
loop is cheaper on the CPU. Shipping it to the GPU would make your program slower.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
