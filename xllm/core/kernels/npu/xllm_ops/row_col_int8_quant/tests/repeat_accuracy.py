import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--solution", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--entry", choices=("run", "run_direct"), default="run")
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    sys.path.insert(0, str(args.spec.resolve()))
    import generate_data
    import golden
    import verify
    s = importlib.util.spec_from_file_location("solution", args.solution)
    sol = importlib.util.module_from_spec(s)
    s.loader.exec_module(sol)
    torch.npu.set_device(0)
    report = {"status": "running", "entry": args.entry, "npu_executed": True, "records": []}
    try:
        for case in json.loads((args.spec / "case.json").read_text()):
            arrays = generate_data.generate(case)
            xs = verify.device_inputs(case, arrays, "npu:0")
            expected = golden.calculate(arrays, {})
            for iteration in range(args.repeats):
                ys = getattr(sol, args.entry)(xs, {})
                torch.npu.synchronize()
                verify.validate_ownership(xs, ys)
                checks = verify.compare(case, arrays, ys, expected)
                assert np.array_equal(verify.cpu_array(xs[0]).view(np.uint16), arrays[0].view(np.uint16))
                del ys
                report["records"].append({"case_name": case["case_name"], "iteration": iteration,
                                          "status": "passed", "outputs": checks})
            print(case["case_name"], "passed", args.repeats, flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", active_case=case["case_name"], iteration=iteration, error=repr(exc))
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
