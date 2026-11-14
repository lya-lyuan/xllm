# RowColInt8Quant

Ascend C implementation for the independent bidirectional INT8 quantization
task. Development target: `openlibing-DevEnv-109476`, logical NPU 0,
Ascend910_9382, CANN 9.1.0.

## Build

```bash
cd /workspace/int8-rowcol-release
mkdir -p logs
/usr/local/python3.12.13/bin/python3.12 -m venv --system-site-packages .venv
ROWCOL_WHEELS=${ROWCOL_WHEELS:-/workspace}
.venv/bin/python3.12 -m pip install --no-index --no-deps \
  "$ROWCOL_WHEELS"/numpy-2.2.6-*aarch64*.whl \
  "$ROWCOL_WHEELS"/ml_dtypes-0.6.0-*aarch64*.whl
bash build_all.sh > logs/build.log 2>&1
source env.sh
./build_examples/aclnn_example
```

The generated `.run` package installs beneath
`$HOME/.local/rowcol-int8`, never into system CANN. Set `ROWCOL_PREFIX` to
choose a different private installation directory. The binding is built
ahead of testing; the solution does not JIT compile during timed calls.
`env.sh` selects a project-local `.venv` when present. The release validation
uses a private venv with NumPy 2.2.6 and ml_dtypes 0.6.0, while reusing the
matching system torch and torch_npu packages. Set `ROWCOL_WHEELS` to the supplied
wheelhouse when using another directory. Report generation additionally needs
openpyxl 3.1.5. Do not upgrade shared Python packages for other tasks.

## Interfaces

```python
import solution
outputs = solution.run([x], {})         # formal two-stage ACLNN API
outputs = solution.run_direct([x], {})  # shared kernel, three ordered stages
call = solution.prepare([x], {})
outputs = call()                        # fresh independent outputs each time
```

Non-NPU, non-FP16, non-contiguous, invalid shape or unknown attributes are
errors. There is deliberately no CPU compute fallback.

## Reproduce

Keep the original public task files in `/workspace/int8-task-spec`.

```bash
source env.sh
python3.12 /workspace/int8-task-spec/verify.py --solution-file "$PWD/solution.py" --report reports/public_accuracy.json
python3.12 tests/repeat_accuracy.py --spec /workspace/int8-task-spec --solution "$PWD/solution.py" --report reports/repeat.json
python3.12 tests/validate_extended.py --spec /workspace/int8-task-spec --solution "$PWD/solution.py" --report reports/extended.json
python3.12 tests/validate_memory.py --spec /workspace/int8-task-spec --solution "$PWD/solution.py" --report reports/memory.json
python3.12 /workspace/int8-task-spec/benchmark.py --solution-file "$PWD/solution.py" --report reports/performance.json
python3.12 /workspace/int8-task-spec/check_performance.py --report reports/performance.json --output reports/performance_acceptance.json
python3.12 tests/benchmark_baselines.py --spec /workspace/int8-task-spec --report reports/baselines.json
```

See [the design](docs/design.md) for indexing, buffer budgets and numerical
semantics. JSON reports and raw logs are the evidence, not this README.
Historical failed/interrupted trials are retained separately from final runs.

## Environment Boundary

The supplied DevEnv has Ubuntu 22.04.5 and driver 26.1.1, whereas the task lists
openEuler 24.03 LTS-SP3 and driver 25.5.t6.b001. CANN, Python and framework
versions must be checked in `reports/environment.json`. Normal pip index
resolution for ml_dtypes 0.6.0 failed, so its aarch64 wheel was selected from
the official PyPI release JSON and verified against the published SHA256:
`37da32aa97749251025666d62372775019594577b9c9e9cfda83bed48d778fdb`.
Full revalidation results on this DevEnv do not imply that the OS/driver
differences have been approved by the external acceptance party. Community
design review and publication are separate from implementation and local NPU
test results.
