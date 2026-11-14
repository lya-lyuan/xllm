# RowColInt8Quant

Ascend C implementation for the independent bidirectional INT8 quantization
task. Development target: `openlibing-DevEnv-109476`, logical NPU 0,
Ascend910_9382, CANN 9.1.0.

## Build

```bash
mkdir -p logs
/usr/local/python3.12.13/bin/python3.12 -m venv --system-site-packages .venv
.venv/bin/python3.12 -m pip install --no-deps 'numpy==2.2.6' \
  'https://files.pythonhosted.org/packages/ed/cf/87e8a6c57eed63a91782a0d229856ddf73e138ce004dd71e2799a9dcdb33/ml_dtypes-0.6.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=37da32aa97749251025666d62372775019594577b9c9e9cfda83bed48d778fdb'
bash build_all.sh > logs/build.log 2>&1
source env.sh
./build_examples/aclnn_example
```

The generated `.run` package installs beneath
`$HOME/.local/rowcol-int8`, never into system CANN. Set `ROWCOL_PREFIX` to
choose a different private installation directory. The binding is built
ahead of testing; the solution does not JIT compile during timed calls.
The installer reads the actual package name from `build_out/CPackConfig.cmake`,
so it does not assume an Ubuntu package name on openEuler. A missing package or
an installer error terminates the build instead of selecting a stale package.
`env.sh` selects a project-local `.venv` when present. The release validation
uses a private venv with NumPy 2.2.6 and ml_dtypes 0.6.0, while reusing the
matching system torch and torch_npu packages. The commands download dependencies
from public package hosts; no local wheelhouse attachment is required. Run from
this operator project directory in a fresh checkout. Report generation needs
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

The unchanged public task files are included in `tests/public_task.zip`.
See [their provenance](tests/PUBLIC_TASK.md). No separate local task attachment
or `/workspace/int8-task-spec` directory is required for a current checkout.

```bash
source env.sh
python3.12 -m zipfile -e tests/public_task.zip tests/public_task
export TASK_ROOT="$PWD/tests/public_task"
mkdir -p reports logs
python3.12 "$TASK_ROOT/verify.py" --solution-file "$PWD/solution.py" --report reports/public_accuracy.json
python3.12 tests/repeat_accuracy.py --spec "$TASK_ROOT" --solution "$PWD/solution.py" --report reports/repeat.json
python3.12 tests/validate_extended.py --spec "$TASK_ROOT" --solution "$PWD/solution.py" --report reports/extended.json
python3.12 tests/validate_memory.py --spec "$TASK_ROOT" --solution "$PWD/solution.py" --report reports/memory.json
python3.12 "$TASK_ROOT/benchmark.py" --solution-file "$PWD/solution.py" --report reports/performance.json
python3.12 "$TASK_ROOT/check_performance.py" --report reports/performance.json --output reports/performance_acceptance.json
python3.12 tests/benchmark_baselines.py --spec "$TASK_ROOT" --report reports/baselines.json
```

See [the design](docs/design.md) for indexing, buffer budgets and numerical
semantics. JSON reports and raw logs are the evidence, not this README.
Historical failed/interrupted trials are retained separately from final runs.

## Delivery Tool Regressions

With CMake and openpyxl 3.1.5 available, run the CPU-only regressions:

```bash
python3.12 -m unittest discover -s tests -p test_delivery_tools.py -v
```

Report generation requires successful public, extended, repeated-accuracy,
memory, performance and baseline evidence, including per-case coverage.
At least one precision-qualified baseline must exist per performance case;
both prescribed candidates must be considered and exclusions explained.
The selected baseline must be the fastest qualified one. Performance checks
are recomputed by the original public checker. Missing or failed gates stop
packaging with a nonzero exit status; an earlier archive is not revalidated by
a failed packaging invocation. The seven materials remain a local submission.

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
