#!/usr/bin/env bash
# 编译 source-compiled 语料：Rust 与 C × 三个优化级别（规格 §61 / §62）。
#
# 用法: bash corpus/source-compiled/build.sh
# 产物: corpus/source-compiled/out/<lang>-o<level>.wasm
#
# 缺少工具链时报错并以非零退出，不静默跳过 —— 静默跳过会让「语料齐了」变成
# 一个无法证伪的声明。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
mkdir -p "$OUT"

# macOS 自带的 Apple clang 不含 WebAssembly 后端，Homebrew 装的完整 LLVM 才有。
# 它的 bin 不在默认 PATH 里，因此显式前置；wasm-ld 也在同一目录，clang 会自己找到。
for candidate in /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin; do
  if [[ -x "$candidate/clang" ]]; then
    export PATH="$candidate:$PATH"
    break
  fi
done

LEVELS=("0" "1" "3")
BUILT=()
MISSING=()

echo "══ Rust 语料 ═══════════════════════════════════════"
if rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  for level in "${LEVELS[@]}"; do
    target="$OUT/rust-o$level.wasm"
    if rustc --target wasm32-unknown-unknown --crate-type cdylib \
        -C opt-level="$level" -C debuginfo=0 -C panic=abort \
        -o "$target" "$HERE/rust/pointwise.rs" 2>"$OUT/.rust-o$level.err"; then
      echo "  ✔ rust-o$level.wasm ($(wc -c < "$target" | tr -d ' ') 字节)"
      BUILT+=("rust-o$level")
    else
      echo "  ✖ rust-o$level 编译失败:"
      sed 's/^/      /' "$OUT/.rust-o$level.err" | head -5
      MISSING+=("rust-o$level")
    fi
    rm -f "$OUT/.rust-o$level.err"
  done
else
  echo "  ✖ 缺少 wasm32-unknown-unknown target"
  echo "      修复: rustup target add wasm32-unknown-unknown"
  MISSING+=("rust-o0" "rust-o1" "rust-o3")
fi

echo ""
echo "══ C 语料 ══════════════════════════════════════════"
probe_source="$OUT/.probe.c"
probe_output="$OUT/.probe.wasm"
printf 'int probe(void) { return 0; }\n' > "$probe_source"
if clang --target=wasm32 -nostdlib -Wl,--no-entry \
    -o "$probe_output" "$probe_source" >/dev/null 2>&1; then
  for level in "${LEVELS[@]}"; do
    target="$OUT/c-o$level.wasm"
    if clang --target=wasm32 -nostdlib -Wl,--no-entry \
        -Wl,--export-all -O"$level" \
        -o "$target" "$HERE/c/pointwise.c" 2>"$OUT/.c-o$level.err"; then
      echo "  ✔ c-o$level.wasm ($(wc -c < "$target" | tr -d ' ') 字节)"
      BUILT+=("c-o$level")
    else
      echo "  ✖ c-o$level 编译失败:"
      sed 's/^/      /' "$OUT/.c-o$level.err" | head -5
      MISSING+=("c-o$level")
    fi
    rm -f "$OUT/.c-o$level.err"
  done
else
  echo "  ✖ 当前 clang 不支持 wasm32 target（或缺少 wasm-ld）"
  echo "      实测: $(clang --version 2>&1 | head -1)"
  echo "      修复: 安装含 WebAssembly 后端的完整 LLVM（如 brew install llvm）"
  echo "      注意: macOS 自带的 Apple clang 不含该后端"
  MISSING+=("c-o0" "c-o1" "c-o3")
fi
rm -f "$probe_source" "$probe_output"

echo ""
echo "════════════════════════════════════════════════════"
echo "已生成 ${#BUILT[@]} 个语料: ${BUILT[*]:-无}"
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "未生成 ${#MISSING[@]} 个语料: ${MISSING[*]}"
  exit 1
fi
echo "全部语料已生成"
