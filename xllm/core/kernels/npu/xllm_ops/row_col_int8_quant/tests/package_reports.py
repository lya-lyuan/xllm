"""Populate the official seven-file template from real, final JSON/log evidence."""
import argparse
from copy import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import openpyxl
from openpyxl.styles import Alignment, PatternFill


def load(path):
    return json.loads(path.read_text())


def workbook(template, prefix):
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


def row(sheet, index, values):
    for column, value in enumerate(values, 1):
        cell = sheet.cell(index, column)
        if index != 4:
            cell._style = copy(sheet.cell(4, column)._style)
        cell.value = value
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        if value in ("passed", "failed"):
            cell.fill = PatternFill("solid", fgColor="D8EAD2" if value == "passed" else "F4CCCC")
    sheet.row_dimensions[index].height = 48


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    args = p.parse_args()
    root, reports = args.project, args.project / "reports"
    public = load(reports / "public_accuracy.json")
    extended = load(reports / "extended.json")
    memory = load(reports / "memory.json")
    performance = load(reports / "performance.json")
    acceptance = load(reports / "performance_acceptance.json")
    baselines = load(reports / "baselines.json")
    env = load(reports / "environment.json")
    coverage = load(args.spec / "coverage_requirements.json")
    definitions = {c["case_name"]: c for c in load(args.spec / "case.json")}
    targets = load(args.spec / "performance_targets.json")
    targets_by_id = {item["case_name"]: item for item in targets["cases"]}
    out = root / "task_submission"
    out.mkdir(exist_ok=True)
    actual = {(c["case_name"], "run"): c["status"] for c in public["cases"]}
    actual.update({(r["check"], r["entry"]): r["status"] for r in extended["records"]})
    required_valid = {(name, entry) for name in coverage["public_cases"] for entry in ("run", "run_direct")}
    required_valid |= {(name, entry) for name in coverage["extra_input_case_ids"] for entry in coverage["extra_entries"]}
    required_negative = {(name, entry) for name in coverage["negative_scenarios"] for entry in coverage["negative_entries"]}
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
        "performance_status": acceptance["status"], "memory_status": memory["status"],
        "environment_acceptance": "pending: compare recorded DevEnv versions with exact task environment",
        "community_review_and_publication": "pending; no external review or repository submission claimed",
    }
    (reports / "delivery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    environment_note = (f"Device {env['device']}; Python {env['python']}; packages {env['versions']}; "
                        f"allocator {env['environment']['PYTORCH_NPU_ALLOC_CONF']}. "
                        "Actual OS/driver may differ from the task; see environment.json. "
                        "Local checks do not constitute external acceptance.")
    with zipfile.ZipFile(args.template) as template:
        name, book, sheet = workbook(template, "2.1 ")
        sheet["A1"] = (f"Accuracy: {summary['accuracy_status']}. 39 distinct valid inputs, "
                       "78 input/entry records; 13 invalid scenarios, 39 entry rejection records. "
                       "An additional physical-format rejection regression is recorded separately. "
                       "Repeated calls do not add independent inputs. " + environment_note)
        checks = [(r["case_name"], "run", r["status"], r.get("outputs", [])) for r in public["cases"]]
        checks += [(r["check"], r["entry"], r["status"], r.get("details", r.get("error")))
                   for r in extended["records"] if r["check"].startswith("RowColInt8Quant_")]
        for i, (case, entry, status, detail) in enumerate(checks, 4):
            shape = definitions[case.split("__extra_seed_")[0]]["input_desc"][0]["shape"]
            row(sheet, i, [f"{case} / {entry}: {shape}", "float16", status,
                           "2.2 accuracy log; reports/public_accuracy.json, extended.json",
                           json.dumps(detail, ensure_ascii=False)])
        engineering = book.create_sheet("Engineering")
        engineering.append(["Check", "Entry", "Status", "Details"])
        for r in extended["records"]:
            if not r["check"].startswith("RowColInt8Quant_"):
                engineering.append([r["check"], r["entry"], r["status"], json.dumps(r, ensure_ascii=False)])
        book.save(out / name)

        name, book, sheet = workbook(template, "3.1 ")
        sheet["A1"] = (f"Performance: {acceptance['status']}. Actual formal run() calls with new outputs; "
                       "50 warmups, 5 rounds x 200 calls, median wall time. Device events include host gaps. "
                       "See raw rounds and excluded baseline reasons. " + environment_note)
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
            status = perf_check.get("status", "passed" if measured["wall_us_median"] <= targets_by_id[case]["max_wall_us"] else "failed")
            row(sheet, i, [str(definitions[case]["input_desc"][0]["shape"]), "float16", status,
                           "3.2 performance log", fastest, measured["wall_us_median"],
                           measured["wall_us_median"] / fastest if fastest else None,
                           json.dumps({"case": case, "selected_baseline": baseline.get("selected"),
                                       "checker": perf_check, "target_definition": targets_by_id[case]}, ensure_ascii=False)])
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

        name, book, sheet = workbook(template, "4.1 ")
        sheet["A1"] = (f"Memory: {memory['status']}. Active NPU allocator bytes, not device HBM. "
                       "5 input shapes x 2 entries; warmed runtime, input already allocated; "
                       "output + workspace + 16 KiB allowance; 20-call reclamation. " + environment_note)
        sheet["E3"], sheet["F3"], sheet["G3"] = "Allowed active peak delta (GB)", "Actual active peak delta (GB)", "Actual/Allowed"
        for i, item in enumerate(memory["cases"], 4):
            limit = item["output_bytes"] + item["workspace_bound_bytes"] + item["alignment_allowance_bytes"]
            row(sheet, i, [f"{item['case_name']} / {item['entry']}", "float16", item["status"],
                           "4.2 memory log", limit / 1e9, item.get("peak_delta_bytes", 0) / 1e9,
                           item.get("peak_delta_bytes", 0) / limit, json.dumps(item)])
        book.save(out / name)

        log_sets = {"2.2 ": ["public_accuracy.log", "repeat.log", "extended.log", "cpp-example.log"],
                    "3.2 ": ["performance.log", "baselines.log", "performance-check.log"],
                    "4.2 ": ["memory.log", "environment.log"]}
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
            "## Reproduce\n\nSee ../README.md for build, private install and exact validation commands. "
            "Original public task files were not modified. The original SHA256SUMS task-document "
            "filename has an extra prefix; case.json and test script hashes match.\n\n"
            "## Evidence\n\nThe three workbooks use the official template and contain real records only. "
            "The three logs combine final raw logs. Full JSON, source/build SHA256 and actual versions "
            "are in ../reports; algorithm and UB/workspace budgets are in ../docs/design.md. "
            "No community design review, PR, exact-environment certification or maintainer acceptance "
            "is implied by these local results.\n")
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
    print(json.dumps({"archive": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "materials": [p.name for p in files], "status": summary}, indent=2))


if __name__ == "__main__":
    main()
