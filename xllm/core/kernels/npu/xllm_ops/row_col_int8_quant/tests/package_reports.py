# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Populate the official seven-file template from real, final JSON/log evidence."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import statistics
import zipfile
from copy import copy
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, PatternFill
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet


def _validate_reports(reports: dict[str, Any], coverage: dict[str, Any]) -> None:
    for name, report in reports.items():
        if report.get("status") != "passed" or report.get("npu_executed") is not True:
            raise ValueError(f"{name}: successful NPU evidence is required")
    for name, field in (("public", "cases"), ("extended", "records"), ("repeat", "records"), ("memory", "cases")):
        if not reports[name][field] or any(record.get("status") != "passed" for record in reports[name][field]):
            raise ValueError(f"{name}: missing or failed per-case evidence")

    actual = {(record["case_name"], "run") for record in reports["public"]["cases"]}
    actual.update((record["check"], record["entry"]) for record in reports["extended"]["records"])
    required = {(name, entry) for name in coverage["public_cases"] for entry in ("run", "run_direct")}
    required.update((name, entry) for name in coverage["extra_input_case_ids"] for entry in coverage["extra_entries"])
    required.update((name, entry) for name in coverage["negative_scenarios"] for entry in coverage["negative_entries"])
    required.update(
        (name, entry)
        for entry in ("run", "run_direct")
        for name in ("repeated_call_reclaim", "output_lifetime_and_input_release", "cross_stream_early_input_release")
    )
    required.add(("two_streams", "prepare/aclnn"))
    required.update(
        (name, "verifier_negative_control")
        for name in ("cpu_outputs", "input_storage_alias", "overlapping_output_storage")
    )
    if required - actual:
        raise ValueError(f"Missing required input/engineering records: {sorted(required - actual)}")
    metadata = {record["check"] for record in reports["extended"]["records"] if record["entry"] == "cpp_metadata"}
    if len(metadata) < coverage["min_engineering_metadata_checks"]:
        raise ValueError("Insufficient compiled metadata rejection checks")

    repeat = reports["repeat"]
    if repeat.get("entry") not in ("run", "run_direct"):
        raise ValueError("Repeated accuracy must identify its tested entry")
    repeated = {(record["case_name"], record["iteration"]) for record in repeat["records"]}
    if not {(name, index) for name in coverage["public_cases"] for index in range(20)} <= repeated:
        raise ValueError("Missing repeated public accuracy checks")

    performance_ids = set(coverage["performance_case_ids"])
    memory = {(record["case_name"], record["entry"]) for record in reports["memory"]["cases"]}
    if memory != {(name, entry) for name in performance_ids for entry in ("run", "run_direct")}:
        raise ValueError("Incomplete memory case/entry coverage")
    baselines = reports["baselines"]
    if {record["case_name"] for record in baselines["cases"]} != performance_ids:
        raise ValueError("Incomplete baseline case coverage")
    if baselines["protocol"] != reports["performance"]["protocol"]:
        raise ValueError("Baseline and submission timing protocols differ")
    for case in baselines["cases"]:
        if case.get("status") != "passed":
            raise ValueError(f"{case['case_name']}: baseline qualification failed")
        candidates = case["candidates"]
        if set(candidates) != {"fp32_absmax_round_cast", "two_dynamic_quant_transpose"}:
            raise ValueError(f"{case['case_name']}: both prescribed baselines are required")
        qualified = {}
        for name, candidate in candidates.items():
            if candidate.get("qualified") is not True:
                if candidate.get("qualified") is not False or not candidate.get("exclusion_reason"):
                    raise ValueError(f"{case['case_name']}/{name}: missing exclusion reason")
                continue
            if not candidate.get("accuracy"):
                raise ValueError(f"{case['case_name']}/{name}: missing precision qualification")
            timing = candidate["timing"]
            for field in ("wall_us_rounds", "device_event_us_rounds"):
                values = timing[field]
                if len(values) != 5 or any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError(f"{case['case_name']}/{name}: invalid timing rounds")
            median = statistics.median(timing["wall_us_rounds"])
            if timing["wall_us_median"] != median:
                raise ValueError(f"{case['case_name']}/{name}: inconsistent timing median")
            qualified[name] = median
        if not qualified:
            raise ValueError(f"{case['case_name']}: no precision-qualified baseline")
        selected = case.get("selected")
        fastest = min(qualified.values())
        if (
            selected not in qualified
            or qualified[selected] != fastest
            or case.get("selected_wall_us_median") != fastest
        ):
            raise ValueError(f"{case['case_name']}: selected baseline is not the fastest qualified candidate")


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _workbook(template: zipfile.ZipFile, prefix: str) -> tuple[str, Workbook, Worksheet]:
    name = next(n for n in template.namelist() if n.startswith(prefix) and n.endswith(".xlsx"))
    book = openpyxl.load_workbook(io.BytesIO(template.read(name)))
    for page in book:
        page._images.clear()
        page._charts.clear()
    sheet = book.active
    for row in sheet.iter_rows(min_row=4):
        for cell in row:
            if cell.__class__.__name__ != "MergedCell":
                cell.value = None
    return name, book, sheet


def _write_row(sheet: Worksheet, index: int, values: list[Any]) -> None:
    for column, value in enumerate(values, 1):
        cell = sheet.cell(index, column)
        if index != 4:
            cell._style = copy(sheet.cell(4, column)._style)
        cell.value = value
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        if value in ("passed", "failed"):
            cell.fill = PatternFill("solid", fgColor="D8EAD2" if value == "passed" else "F4CCCC")
    sheet.row_dimensions[index].height = 48


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    args = p.parse_args()
    root, reports = args.project, args.project / "reports"
    public = _load(reports / "public_accuracy.json")
    extended = _load(reports / "extended.json")
    repeat = _load(reports / "repeat.json")
    memory = _load(reports / "memory.json")
    performance = _load(reports / "performance.json")
    acceptance = _load(reports / "performance_acceptance.json")
    baselines = _load(reports / "baselines.json")
    env = _load(reports / "environment.json")
    coverage = _load(args.spec / "coverage_requirements.json")
    definitions = {c["case_name"]: c for c in _load(args.spec / "case.json")}
    targets = _load(args.spec / "performance_targets.json")
    targets_by_id = {item["case_name"]: item for item in targets["cases"]}
    _validate_reports(
        {
            "public": public,
            "extended": extended,
            "repeat": repeat,
            "memory": memory,
            "performance": performance,
            "baselines": baselines,
        },
        coverage,
    )
    checker_spec = importlib.util.spec_from_file_location(
        "rowcol_performance_checker", args.spec / "check_performance.py"
    )
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    recomputed = checker.evaluate(performance, targets, list(definitions.values()))
    if recomputed["status"] != "passed" or acceptance != recomputed:
        raise ValueError("Performance acceptance evidence is failed or inconsistent with measured rounds")
    cpp_log = (root / "logs/cpp-example.log").read_text()
    if not any(line.startswith("ACLNN example passed; workspace_bytes=") for line in cpp_log.splitlines()):
        raise ValueError("Missing successful independent C++ example evidence")
    out = root / "task_submission"
    out.mkdir(exist_ok=True)
    actual = {(c["case_name"], "run"): c["status"] for c in public["cases"]}
    actual.update({(r["check"], r["entry"]): r["status"] for r in extended["records"]})
    required_valid = {(name, entry) for name in coverage["public_cases"] for entry in ("run", "run_direct")}
    required_valid |= {
        (name, entry) for name in coverage["extra_input_case_ids"] for entry in coverage["extra_entries"]
    }
    required_negative = {
        (name, entry) for name in coverage["negative_scenarios"] for entry in coverage["negative_entries"]
    }
    missing_or_failed = sorted(k for k in required_valid | required_negative if actual.get(k) != "passed")
    precision_pass = not missing_or_failed and public["status"] == extended["status"] == "passed"
    summary = {
        "accuracy_status": "passed" if precision_pass else "failed",
        "independent_valid_inputs": len(coverage["public_cases"]) + len(coverage["extra_input_case_ids"]),
        "independent_invalid_scenarios": len(coverage["negative_scenarios"]),
        "valid_input_entry_records_required": len(required_valid),
        "invalid_entry_records_required": len(required_negative),
        "additional_negative_scenarios": ["physical_format"],
        "missing_or_failed": missing_or_failed,
        "performance_status": acceptance["status"],
        "memory_status": memory["status"],
        "repeat_status": repeat["status"],
        "baseline_status": baselines["status"],
        "environment_acceptance": "pending: compare recorded DevEnv versions with exact task environment",
        "community_review_and_publication": "not evaluated by this report generator; see the delivery index",
        "source_revision": env["source_revision"],
    }
    (reports / "delivery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    environment_note = (
        f"Device {env['device']}; Python {env['python']}; packages {env['versions']}; "
        f"allocator {env['environment']['PYTORCH_NPU_ALLOC_CONF']}. "
        "Actual OS/driver may differ from the task; see environment.json. "
        "Local checks do not constitute external acceptance."
    )
    with zipfile.ZipFile(args.template) as template:
        name, book, sheet = _workbook(template, "2.1 ")
        sheet["A1"] = (
            f"Accuracy: {summary['accuracy_status']}. 39 distinct valid inputs, "
            "78 input/entry records; 13 invalid scenarios, 39 entry rejection records. "
            "An additional physical-format rejection regression is recorded separately. "
            "Repeated calls do not add independent inputs. " + environment_note
        )
        checks = [(r["case_name"], "run", r["status"], r.get("outputs", [])) for r in public["cases"]]
        checks += [
            (r["check"], r["entry"], r["status"], r.get("details", r.get("error")))
            for r in extended["records"]
            if r["check"].startswith("RowColInt8Quant_")
        ]
        for i, (case, entry, status, detail) in enumerate(checks, 4):
            shape = definitions[case.split("__extra_seed_")[0]]["input_desc"][0]["shape"]
            _write_row(
                sheet,
                i,
                [
                    f"{case} / {entry}: {shape}",
                    "float16",
                    status,
                    "2.2 accuracy log; reports/public_accuracy.json, extended.json",
                    json.dumps(detail, ensure_ascii=False),
                ],
            )
        engineering = book.create_sheet("Engineering")
        engineering.append(["Check", "Entry", "Status", "Details"])
        for r in extended["records"]:
            if not r["check"].startswith("RowColInt8Quant_"):
                engineering.append([r["check"], r["entry"], r["status"], json.dumps(r, ensure_ascii=False)])
        book.save(out / name)

        name, book, sheet = _workbook(template, "3.1 ")
        sheet["A1"] = (
            f"Performance: {acceptance['status']}. Actual formal run() calls with new outputs; "
            "50 warmups, 5 rounds x 200 calls, median wall time. Device events include host gaps. "
            "See raw rounds and excluded baseline reasons. " + environment_note
        )
        baseline_by_id = {c["case_name"]: c for c in baselines["cases"]}
        rounds = book.create_sheet("Raw Rounds")
        rounds.append(["Case", "Candidate", "Round", "Wall us", "Device-event us"])
        for i, item in enumerate(performance["cases"], 4):
            case = item["case_name"]
            measured = item["results"]["custom"]
            baseline = baseline_by_id[case]
            fastest = baseline.get("selected_wall_us_median")
            # Target schema is preserved as JSON in the explanation; checker owns pass/fail.
            perf_check = next((v for v in acceptance.get("checks", []) if v.get("case_name") == case), {})
            status = perf_check.get(
                "status", "passed" if measured["wall_us_median"] <= targets_by_id[case]["max_wall_us"] else "failed"
            )
            _write_row(
                sheet,
                i,
                [
                    str(definitions[case]["input_desc"][0]["shape"]),
                    "float16",
                    status,
                    "3.2 performance log",
                    fastest,
                    measured["wall_us_median"],
                    measured["wall_us_median"] / fastest if fastest else None,
                    json.dumps(
                        {
                            "case": case,
                            "selected_baseline": baseline.get("selected"),
                            "checker": perf_check,
                            "target_definition": targets_by_id[case],
                        },
                        ensure_ascii=False,
                    ),
                ],
            )
            candidates = {"custom": measured}
            candidates.update({n: v["timing"] for n, v in baseline["candidates"].items() if v["qualified"]})
            for candidate, timing in candidates.items():
                for j, (wall, event) in enumerate(zip(timing["wall_us_rounds"], timing["device_event_us_rounds"]), 1):
                    rounds.append([case, candidate, j, wall, event])
        excluded = book.create_sheet("Baseline Qualification")
        excluded.append(["Case", "Candidate", "Qualified", "Details"])
        for case, value in baseline_by_id.items():
            for candidate, result in value["candidates"].items():
                excluded.append([case, candidate, result["qualified"], json.dumps(result, ensure_ascii=False)])
        book.save(out / name)

        name, book, sheet = _workbook(template, "4.1 ")
        sheet["A1"] = (
            f"Memory: {memory['status']}. Active NPU allocator bytes, not device HBM. "
            "5 input shapes x 2 entries; warmed runtime, input already allocated; "
            "output + workspace + 16 KiB allowance; 20-call reclamation. " + environment_note
        )
        sheet["E3"], sheet["F3"], sheet["G3"] = (
            "Allowed active peak delta (GB)",
            "Actual active peak delta (GB)",
            "Actual/Allowed",
        )
        for i, item in enumerate(memory["cases"], 4):
            limit = item["output_bytes"] + item["workspace_bound_bytes"] + item["alignment_allowance_bytes"]
            _write_row(
                sheet,
                i,
                [
                    f"{item['case_name']} / {item['entry']}",
                    "float16",
                    item["status"],
                    "4.2 memory log",
                    limit / 1e9,
                    item.get("peak_delta_bytes", 0) / 1e9,
                    item.get("peak_delta_bytes", 0) / limit,
                    json.dumps(item),
                ],
            )
        book.save(out / name)

        log_sets = {
            "2.2 ": ["public_accuracy.log", "repeat.log", "extended.log", "cpp-example.log"],
            "3.2 ": ["performance.log", "baselines.log", "performance-check.log"],
            "4.2 ": ["memory.log", "environment.log"],
        }
        for prefix, sources in log_sets.items():
            filename = next(n for n in template.namelist() if n.startswith(prefix))
            text = ""
            for source in sources:
                path = root / "logs" / source
                if not path.is_file():
                    raise FileNotFoundError(path)
                text += f"\n===== {source} =====\n" + path.read_text()
            (out / filename).write_text(text)
        filename = next(n for n in template.namelist() if n.startswith("1 "))
        (out / filename).write_text(
            "# RowColInt8Quant Self-Validation\n\n" + environment_note + "\n\n"
            "## Status\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"
            "## Reproduce\n\nUse the fixed source commit and code repository recorded in the delivery index. "
            "The source_revision entry above preserves the environment collector's original Git query result; "
            "source and binary fingerprints are supplied with the raw reports. Enter the operator project "
            "directory xllm/core/kernels/npu/xllm_ops/row_col_int8_quant/. Keep the unchanged public task "
            "package at /workspace/int8-task-spec and set ROWCOL_WHEELS to the supplied aarch64 wheelhouse. "
            "The matching CANN 9.1.0, Python 3.12.13, torch 2.10.0+cpu and torch_npu 2.10.0.post4 "
            "must be available before building.\n\n"
            "```bash\n"
            "set -e\n"
            "/usr/local/python3.12.13/bin/python3.12 -m venv --system-site-packages .venv\n"
            ".venv/bin/python3.12 -m pip install --no-index --no-deps "
            '"$ROWCOL_WHEELS"/numpy-2.2.6-*aarch64*.whl "$ROWCOL_WHEELS"/ml_dtypes-0.6.0-*aarch64*.whl\n'
            "export ASCEND_RT_VISIBLE_DEVICES=0\n"
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True\n"
            'export ROWCOL_PREFIX="$HOME/.local/rowcol-int8"\n'
            "mkdir -p logs reports\n"
            "bash build_all.sh > logs/build.log 2>&1\n"
            "source env.sh\n"
            "./build_examples/aclnn_example > logs/cpp-example.log 2>&1\n"
            'python3.12 /workspace/int8-task-spec/verify.py --solution-file "$PWD/solution.py" --report reports/public_accuracy.json > logs/public_accuracy.log 2>&1\n'
            'python3.12 tests/repeat_accuracy.py --spec /workspace/int8-task-spec --solution "$PWD/solution.py" --report reports/repeat.json > logs/repeat.log 2>&1\n'
            'python3.12 tests/validate_extended.py --spec /workspace/int8-task-spec --solution "$PWD/solution.py" --report reports/extended.json > logs/extended.log 2>&1\n'
            'python3.12 tests/validate_memory.py --spec /workspace/int8-task-spec --solution "$PWD/solution.py" --report reports/memory.json > logs/memory.log 2>&1\n'
            'python3.12 /workspace/int8-task-spec/benchmark.py --solution-file "$PWD/solution.py" --report reports/performance.json > logs/performance.log 2>&1\n'
            "python3.12 /workspace/int8-task-spec/check_performance.py --report reports/performance.json --output reports/performance_acceptance.json > logs/performance-check.log 2>&1\n"
            "python3.12 tests/benchmark_baselines.py --spec /workspace/int8-task-spec --report reports/baselines.json > logs/baselines.log 2>&1\n"
            "python3.12 tests/collect_environment.py > logs/environment.log 2>&1\n"
            "```\n\n"
            "Do not reuse an extension built at another source path. Installation uses a private user prefix, "
            "not system CANN. Stop on any failed command and preserve its logs. "
            "Original public task files were not modified. The original SHA256SUMS task-document "
            "filename has an extra prefix; case.json and test script hashes match.\n\n"
            "## Evidence\n\nThe three workbooks use the official template and contain real records only. "
            "The three logs combine final raw logs. Full JSON, source/build SHA256 and actual versions "
            "are in the accompanying reports directory; algorithm and UB/workspace budgets are in "
            "the source project's docs/design.md. These attachments are not necessarily inside the seven-file ZIP. "
            "No community design review, PR, exact-environment certification or maintainer acceptance "
            "is implied by these local results.\n"
        )
    files = sorted(p for p in out.iterdir() if p.is_file())
    if len(files) != 7:
        raise ValueError(f"expected exactly 7 official materials, found {len(files)}")
    archive = root / "task_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "materials": [p.name for p in files],
                "status": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
