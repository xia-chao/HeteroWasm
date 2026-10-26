# 贡献指南

## 提交前

- `cargo fmt` 与 `cargo clippy --all-targets` 必须干净
- 改动涉及 GPU 路径时，附上 Metal 上的实测数字（设备、形状、耗时）
- 一个 PR 只做一件事，便于 review 和回滚

## 提交信息

首行用 `模块: 动词短语`，例如 `codegen: 修正 mask 合并的边界`。
正文写清"为什么"，代码本身说明"做了什么"。

## 验证

CPU 路径始终是正确性基准；GPU 路径的加速必须能被 `corpus/` 里的用例复现。
