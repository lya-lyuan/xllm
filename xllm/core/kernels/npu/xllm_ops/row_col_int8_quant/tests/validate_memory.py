"""Active PyTorch NPU allocator accounting, not device-wide HBM usage."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

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
    spec = importlib.util.spec_from_file_location("solution", args.solution)
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    torch.npu.set_device(0)
    report = {"status": "running", "npu_executed": True, "metric": "memory_allocated/max_memory_allocated", "cases": []}
    for case in json.loads((args.spec / "case.json").read_text()):
        if not case["performance"]:
            continue
        arrays = generate_data.generate(case)
        xs = verify.device_inputs(case, arrays, "npu:0")
        m, n = xs[0].shape
        output_bytes = 2 * m * n + 4 * (m + n)
        workspace_bound = 16 * 1024 * 1024 + 4 * ((m + 63) // 64) * ((n + 255) // 256) * 320
        for name in ("run", "run_direct"):
            entry = getattr(sol, name)
            record = {"case_name": case["case_name"], "entry": name,
                      "output_bytes": output_bytes, "workspace_bound_bytes": workspace_bound,
                      "alignment_allowance_bytes": 16 * 1024}
            try:
                for _ in range(3):
                    ys = entry(xs, {})
                    del ys
                torch.npu.synchronize()
                gc.collect()
                before = torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                ys = entry(xs, {})
                peak = torch.npu.max_memory_allocated()
                torch.npu.synchronize()
                verify.validate_ownership(xs, ys)
                verify.compare(case, arrays, ys, golden.calculate(arrays, {}))
                del ys
                torch.npu.synchronize()
                for _ in range(20):
                    ys = entry(xs, {})
                    del ys
                torch.npu.synchronize()
                gc.collect()
                after = torch.npu.memory_allocated()
                record.update(before_bytes=before, peak_bytes=peak, peak_delta_bytes=peak - before,
                              after_bytes=after, repeated_calls=20)
                assert peak - before <= output_bytes + workspace_bound + 16 * 1024, record
                assert before == after, record
                record["status"] = "passed"
            except Exception as exc:
                record.update(status="failed", error=repr(exc))
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
        del xs
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
