"""Extended engineering acceptance. Uses the untouched public generator/verifier."""
import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import sys
import weakref

import numpy as np
import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.spec.resolve()))
    import generate_data
    import golden
    import verify

    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location("solution", args.solution)
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    cases = json.loads((args.spec / "case.json").read_text())
    report = {"status": "running", "npu_executed": True, "records": []}

    def record(name, entry, fn):
        try:
            details = fn()
            item = {"check": name, "entry": entry, "status": "passed", "details": details}
        except Exception as exc:
            item = {"check": name, "entry": entry, "status": "failed", "error": repr(exc)}
        report["records"].append(item)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(item), flush=True)

    def checked(case, entry, offset=False):
        arrays = generate_data.generate(case)
        xs = verify.device_inputs(case, arrays, "npu:0")
        if offset:
            base = torch.empty(xs[0].numel() + 16, dtype=xs[0].dtype, device=xs[0].device)
            x = base[1:1 + xs[0].numel()].view(xs[0].shape)
            x.copy_(xs[0])
            xs = [x]
        ys = entry(xs, {})
        torch.npu.synchronize()
        verify.validate_ownership(xs, ys)
        result = verify.compare(case, arrays, ys, golden.calculate(arrays, {}))
        assert np.array_equal(verify.cpu_array(xs[0]).view(np.uint16), arrays[0].view(np.uint16))
        return result

    entries = {"run": sol.run, "run_direct": sol.run_direct}
    for case in cases:
        record(case["case_name"], "run_direct", lambda c=case: checked(c, sol.run_direct))
    extra = [(0, 0), (1, 1), (2, 2), (2, 6), (2, 7), (3, 3), (4, 4), (5, 5),
             (8, 8), (9, 9), (10, 10), (11, 11)]
    for index, seed in extra:
        case = copy.deepcopy(cases[index])
        case.update(case_name=case["case_name"] + f"__extra_seed_{seed}", kind="random", performance=False)
        for name, entry in entries.items():
            record(case["case_name"], name, lambda c=case, fn=entry: checked(c, fn, offset=True))

    valid = verify.device_inputs(cases[4], generate_data.generate(cases[4]), "npu:0")[0]
    large_m = torch.zeros((4097, valid.size(1)), dtype=valid.dtype, device=valid.device)
    large_m[:valid.size(0)].copy_(valid)
    large_n = torch.zeros((valid.size(0), 4098), dtype=valid.dtype, device=valid.device)
    large_n[:, :valid.size(1)].copy_(valid)
    noncontiguous = torch.empty((valid.size(0), valid.size(1) * 2), dtype=valid.dtype,
                                device=valid.device)[:, ::2]
    noncontiguous.copy_(valid)
    invalid = {
        "attributes_type": ([valid], []), "unknown_attribute": ([valid], {"unknown": 1}),
        "bf16": ([valid.bfloat16()], {}), "float32": ([valid.float()], {}),
        "cpu_input": ([valid.cpu()], {}), "input_count": ([], {}),
        "empty_m": ([valid[:0]], {}), "empty_n": ([valid[:, :0]], {}),
        "large_m": ([large_m], {}), "large_n": ([large_n], {}),
        "noncontiguous": ([noncontiguous], {}),
        "odd_n": ([valid[:, :3].contiguous()], {}),
        "rank": ([valid.reshape(-1)[:8].reshape(2, 2, 2)], {}),
    }
    for entry_name, entry in {**entries, "prepare": sol.prepare}.items():
        for name, (xs, attrs) in invalid.items():
            def rejected(fn=entry, x=xs, a=attrs):
                try:
                    fn(x, a)
                except (RuntimeError, ValueError, TypeError) as exc:
                    return {"error": str(exc), "attributes_type": type(a).__name__,
                            "inputs": [{"shape": list(t.shape), "strides": list(t.stride()),
                                        "dtype": str(t.dtype), "device": str(t.device)} for t in x]}
                raise AssertionError("invalid metadata accepted")
            record(name, entry_name, rejected)

    for name in ("bf16", "float32", "empty_m", "empty_n", "large_m", "large_n", "noncontiguous", "odd_n", "rank"):
        def rejected_cpp(x=invalid[name][0][0]):
            try:
                sol._backend().validate(x)
            except RuntimeError as exc:
                return str(exc)
            raise AssertionError("C++ binding accepted invalid metadata")
        record(name, "cpp_metadata", rejected_cpp)

    old_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    try:
        torch.npu.config.allow_internal_format = True
        blocked = torch_npu.npu_format_cast(torch.ones((32, 32), dtype=valid.dtype, device=valid.device), 29)
        torch.npu.synchronize()
    finally:
        torch.npu.config.allow_internal_format = old_internal_format
    if torch_npu.get_npu_format(blocked) in ("ND", 2):
        raise AssertionError("physical format regression fixture is not FRACTAL_NZ")
    for entry_name, entry in {**entries, "prepare": sol.prepare}.items():
        def rejected_format(fn=entry):
            try:
                fn([blocked], {})
            except (RuntimeError, ValueError, TypeError) as exc:
                return str(exc)
            raise AssertionError("non-ND physical storage accepted")
        record("physical_format", entry_name, rejected_format)

    def verifier_counterexample(kind):
        case = cases[4]
        arrays = generate_data.generate(case)
        xs = verify.device_inputs(case, arrays, "npu:0")
        ys = list(sol.run(xs, {}))
        torch.npu.synchronize()
        if kind == "cpu_outputs":
            ys = [y.cpu() for y in ys]
        elif kind == "input_storage_alias":
            ys[0] = xs[0].view(torch.int8).reshape(-1)[:xs[0].numel()].view(xs[0].shape)
        else:
            count = xs[0].numel()
            base = torch.empty(count + 1, dtype=torch.int8, device=xs[0].device)
            ys[0], ys[1] = base[:count].view(xs[0].shape), base[1:].view(xs[0].shape)
            assert ys[0].data_ptr() != ys[1].data_ptr()
        try:
            verify.validate_ownership(xs, ys)
            verify.compare(case, arrays, ys, golden.calculate(arrays, {}))
        except AssertionError as exc:
            return str(exc)
        raise AssertionError("public verifier accepted a required counterexample")
    for kind in ("cpu_outputs", "input_storage_alias", "overlapping_output_storage"):
        record(kind, "verifier_negative_control", lambda k=kind: verifier_counterexample(k))

    # A queued native matmul workload gives the lifecycle checks a verifiable
    # unfinished device operation without adding a custom delay kernel.
    delay_matrix = torch.ones((4096, 4096), dtype=valid.dtype, device=valid.device)
    delay_output = torch.empty_like(delay_matrix)
    torch.mm(delay_matrix, delay_matrix, out=delay_output)
    torch.npu.synchronize()

    def delay(output, calls=32):
        for _ in range(calls):
            torch.mm(delay_matrix, delay_matrix, out=output)

    def streams():
        fixtures, pending, completed = [], [], []
        for case in (cases[4], cases[8]):
            arrays = generate_data.generate(case)
            xs = verify.device_inputs(case, arrays, "npu:0")
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                stream.wait_stream(torch.npu.default_stream())
                call = sol.prepare(xs, {})
            fixtures.append((case, arrays, xs, stream, call, torch.empty_like(delay_matrix)))
        torch.npu.synchronize()
        for _, _, _, stream, _, output in fixtures:
            with torch.npu.stream(stream):
                delay(output, 16)
        for _ in range(10):
            for case, arrays, xs, stream, call, _ in fixtures:
                with torch.npu.stream(stream):
                    ys = call()
                    pending.append((case, arrays, xs, ys))
        for _, _, _, stream, _, _ in fixtures:
            with torch.npu.stream(stream):
                event = torch.npu.Event()
                event.record()
                completed.append(event)
        unfinished = [not event.query() for event in completed]
        assert all(unfinished), "stream fixture finished before concurrent submissions were observed"
        assert fixtures[0][3] != fixtures[1][3]
        torch.npu.synchronize()
        for case, arrays, xs, ys in pending:
            verify.validate_ownership(xs, ys)
            verify.compare(case, arrays, ys, golden.calculate(arrays, {}))
            assert np.array_equal(verify.cpu_array(xs[0]).view(np.uint16), arrays[0].view(np.uint16))
        return {"streams": 2, "calls_per_stream": 10, "interleaved": True,
                "unfinished_at_submission": unfinished}
    record("two_streams", "prepare/aclnn", streams)

    for name, entry in entries.items():
        def reclaim(fn=entry):
            xs = [valid]
            for _ in range(3):
                ys = fn(xs, {})
                del ys
            torch.npu.synchronize()
            gc.collect()
            before = torch.npu.memory_allocated()
            for _ in range(20):
                ys = fn(xs, {})
                del ys
            torch.npu.synchronize()
            gc.collect()
            after = torch.npu.memory_allocated()
            assert before == after, (before, after)
            return {"before": before, "after": after, "calls": 20}
        record("repeated_call_reclaim", name, reclaim)

        def lifetime(fn=entry):
            arrays = generate_data.generate(cases[4])
            expected = golden.calculate(arrays, {})
            stream = torch.npu.Stream()
            retained = []
            with torch.npu.stream(stream):
                xs = verify.device_inputs(cases[4], arrays, "npu:0")
                for _ in range(8):
                    retained.append(fn(xs, {}))
                for i in range(32):
                    ys = fn(xs, {})
                    del ys
                    for output in retained[0]:
                        torch.empty_like(output).fill_(i)
                del xs
            # A separate stream creates pressure after the last Python reference is released.
            for i in range(32):
                torch.empty_like(valid).fill_(i)
            torch.npu.synchronize()
            storages = [y.untyped_storage().data_ptr() for ys in retained for y in ys]
            assert len(storages) == len(set(storages))
            for ys in retained:
                verify.compare(cases[4], arrays, ys, expected)
            return {"retained_calls": 8, "early_output_release_calls": 32,
                    "matching_output_pressure_allocations": 128}
        record("output_lifetime_and_input_release", name, lifetime)

        def early_input_release(fn=entry):
            case = cases[4]
            arrays = generate_data.generate(case)
            expected = golden.calculate(arrays, {})
            allocator_stream, execution_stream = torch.npu.Stream(), torch.npu.Stream()
            with torch.npu.stream(allocator_stream):
                x = verify.device_inputs(case, arrays, "npu:0")[0]
            allocator_stream.synchronize()
            with torch.npu.stream(execution_stream):
                warm = fn([x], {})
            execution_stream.synchronize()
            del warm
            with torch.npu.stream(execution_stream):
                delay(delay_output)
                ys = fn([x], {})
                done = torch.npu.Event()
                done.record()
            input_storage = x.untyped_storage().data_ptr()
            pending_before_release = not done.query()
            assert pending_before_release, "input-release fixture has no unfinished device work"
            reference = weakref.ref(x)
            del x
            assert reference() is None, "test still owns a Python input reference"
            pending_pressure = 0
            for i in range(32):
                with torch.npu.stream(allocator_stream):
                    pressure = torch.empty(case["input_desc"][0]["shape"], dtype=valid.dtype, device=valid.device)
                    if not done.query():
                        pending_pressure += 1
                        assert pressure.untyped_storage().data_ptr() != input_storage, "input storage recycled while in use"
                    pressure.fill_(i)
                    del pressure
            assert pending_pressure > 0, "all allocation pressure occurred after device completion"
            torch.npu.synchronize()
            verify.compare(case, arrays, ys, expected)
            assert len({y.untyped_storage().data_ptr() for y in ys}) == len(ys)
            assert input_storage not in {y.untyped_storage().data_ptr() for y in ys}
            return {"different_allocation_and_execution_streams": True,
                    "pending_before_input_release": pending_before_release,
                    "python_input_reference_released": True,
                    "pressure_calls": 32, "pressure_allocations_while_pending": pending_pressure}
        record("cross_stream_early_input_release", name, early_input_release)

    report["status"] = "passed" if all(r["status"] == "passed" for r in report["records"]) else "failed"
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
