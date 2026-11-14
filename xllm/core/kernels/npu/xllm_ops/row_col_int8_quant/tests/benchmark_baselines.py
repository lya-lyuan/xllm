"""Measure only candidates qualified by the unmodified public accuracy rules."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.spec.resolve()))
    import benchmark
    import generate_data
    import golden
    import verify
    torch.npu.set_device(0)
    report = {"status": "running", "npu_executed": True, "protocol": benchmark.PROTOCOL, "cases": []}
    for case in json.loads((args.spec / "case.json").read_text()):
        if not case["performance"]:
            continue
        arrays = generate_data.generate(case)
        x = verify.device_inputs(case, arrays, "npu:0")[0]

        def composition():
            a = x.float()
            absolute = a.abs()
            r, c = absolute.amax(dim=1), absolute.amax(dim=0)

            def quant(scale):
                safe = torch.where(scale == 0, torch.ones_like(scale), scale)
                return (a * (127.0 / safe)).round().clamp(-127, 127).to(torch.int8)
            return quant(r[:, None]), quant(c[None, :]), r, c

        def dynamic():
            qr, _ = torch_npu.npu_dynamic_quant(x)
            qc, _ = torch_npu.npu_dynamic_quant(x.t().contiguous())
            absolute = x.float().abs()
            return qr, qc.t().contiguous(), absolute.amax(dim=1), absolute.amax(dim=0)

        record = {"case_name": case["case_name"], "candidates": {}}
        for name, fn in {"fp32_absmax_round_cast": composition, "two_dynamic_quant_transpose": dynamic}.items():
            candidate = {}
            try:
                ys = fn()
                torch.npu.synchronize()
                verify.validate_ownership([x], ys)
                candidate["accuracy"] = verify.compare(case, arrays, ys, golden.calculate(arrays, {}))
                del ys
            except Exception as exc:
                candidate.update(qualified=False, exclusion_reason=repr(exc))
            else:
                candidate.update(qualified=True, timing=benchmark.measure(fn, torch.npu))
            record["candidates"][name] = candidate
        qualified = [(name, c["timing"]["wall_us_median"]) for name, c in record["candidates"].items() if c["qualified"]]
        if qualified:
            record["selected"], record["selected_wall_us_median"] = min(qualified, key=lambda item: item[1])
            record["status"] = "passed"
        else:
            record["status"] = "failed"
        report["cases"].append(record)
        print(json.dumps(record), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
