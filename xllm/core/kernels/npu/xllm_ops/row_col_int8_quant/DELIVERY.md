# Single-Branch RowColInt8Quant Delivery

This directory contains the complete independent operator project. The sibling
`../row_col_int8_quant.cpp` and the declaration in `../xllm_ops_api.h` expose
the xllm C++ entry. No separately published feature branch or unpublished
submodule commit is needed to obtain the new operator source.

## Source Layout

- `op_kernel/`: Ascend C kernel and shared tiling data.
- `op_host/`: OpDef, shape/dtype inference, and tiling.
- `direct/`: direct kernel launch library.
- `python/` and `solution.py`: PyTorch NPU binding and task interfaces.
- `examples/`: independent two-stage ACLNN C++ caller.
- `tests/`: accuracy repeats, invalid metadata, streams, lifetime, memory,
  baseline comparison, and report generation.
- `docs/design.md`: algorithm, numerical semantics, UB and workspace budgets.

The imported standalone source was validated at
`dd87a42a797dc6d8b7a665aa2c6d18a754e79744`; imported C++ files were formatted
with clang-format 20.1.6. The xllm caller was previously validated at
`0db08b6ef64af2bdd8e6ac48bc801665916b817a`. Those historical results do not
by themselves certify the final delivery commit or a complete xllm server.
The former private companion submodule pin is removed from this delivery;
the upstream submodule URL and commit remain unchanged from the main baseline.

## Build

Clone this repository's `feat/rowcol-int8-quant` branch, then enter:

```bash
cd xllm/core/kernels/npu/xllm_ops/row_col_int8_quant
```

Follow `README.md` from the environment-creation step onward, using this
directory instead of the historical `/workspace/int8-rowcol-release` path.
Keep the original public task files separately in `/workspace/int8-task-spec`.
The task package and offline dependency wheelhouse are provided with the local
acceptance ZIP, not implicitly supplied by this Git repository.

`bash build_all.sh` builds the kernel package, installs it under the user's
private prefix, builds the Python/direct entries, and builds the C++ example.
Source `env.sh` before testing and before launching a process that loads the
xllm C++ entry so its ACLNN symbol resolver can find this custom OPP provider.
Do not overwrite system CANN or reuse a prebuilt Python extension from another
source path. Rebuild when relocating this directory.

This standalone project has its own build; it is not automatically built by
the parent xllm CMake source list. The parent list builds the sibling public
caller, which requires the private operator package to be installed first.

## Evidence Boundaries

The seven official report/log materials and full binary/source evidence ZIPs
are kept in the task's local submission directory. Their recorded source and
binary fingerprints must not be silently relabeled as a different commit.
The validated DevEnv uses Ubuntu 22.04.5 / driver 26.1.1 instead of the task's
specified OS/driver image. Community design review, a public upstream PR,
full-server testing and maintainer acceptance remain separate steps.
