# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU-only regressions for package selection and acceptance report gates."""

import copy
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("rowcol_package_reports", Path(__file__).with_name("package_reports.py"))
_REPORTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORTS)


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class PackageInstallationTests(unittest.TestCase):
    def _install(
        self, name: str, *, missing: bool = False, config: bool = True, status: int = 0
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="rowcol install ") as directory:
            root = Path(directory)
            if config:
                (root / "CPackConfig.cmake").write_text(f'set(CPACK_PACKAGE_FILE_NAME "{name}")\n')
            (root / "custom_opp_ubuntu_aarch64.run").write_text("exit 99\n")
            if not missing:
                (root / name).write_text(f"printf '%s\\n' 'selected={name}'\nprintf '%s\\n' \"$@\"\nexit {status}\n")
            result = subprocess.run(
                [
                    "cmake",
                    f"-DBUILD_DIR={root}",
                    f"-DINSTALL_PREFIX={root / 'private prefix'}",
                    "-P",
                    str(_PROJECT / "cmake/install_package.cmake"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.assertIn(f"--install-path={root / 'private prefix'}", result.stdout)
            return result

    def test_actual_cpack_name_on_each_platform(self) -> None:
        for name in ("custom_opp_ubuntu_aarch64.run", "custom_opp_openeuler_aarch64.run", "rowcol_int8.run"):
            with self.subTest(name=name):
                result = self._install(name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"selected={name}", result.stdout)

    def test_missing_package_does_not_install_stale_ubuntu_package(self) -> None:
        result = self._install("custom_opp_openeuler_aarch64.run", missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing generated operator package", result.stderr)

    def test_missing_config_is_rejected(self) -> None:
        result = self._install("custom_opp_openeuler_aarch64.run", config=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing package configuration", result.stderr)

    def test_installer_failure_is_propagated(self) -> None:
        result = self._install("custom_opp_openeuler_aarch64.run", status=17)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installation failed: 17", result.stderr)


class ReportGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coverage = {
            "public_cases": ["public"],
            "extra_input_case_ids": ["extra"],
            "extra_entries": ["run", "run_direct"],
            "negative_scenarios": ["invalid"],
            "negative_entries": ["run", "run_direct", "prepare"],
            "performance_case_ids": ["public"],
            "min_engineering_metadata_checks": 6,
        }
        records = [self._record("public", "run_direct")]
        records += [self._record("extra", entry) for entry in ("run", "run_direct")]
        records += [self._record("invalid", entry) for entry in ("run", "run_direct", "prepare")]
        records += [
            self._record(name, entry)
            for entry in ("run", "run_direct")
            for name in (
                "repeated_call_reclaim",
                "output_lifetime_and_input_release",
                "cross_stream_early_input_release",
            )
        ]
        records += [self._record("two_streams", "prepare/aclnn")]
        records += [
            self._record(name, "verifier_negative_control")
            for name in ("cpu_outputs", "input_storage_alias", "overlapping_output_storage")
        ]
        records += [self._record(f"metadata_{index}", "cpp_metadata") for index in range(6)]
        protocol = {"warmup": 50, "iterations": 200, "rounds": 5}
        candidates = {
            name: {
                "qualified": True,
                "accuracy": [{"output": "qRow"}],
                "timing": {
                    "wall_us_rounds": [value] * 5,
                    "device_event_us_rounds": [value] * 5,
                    "wall_us_median": value,
                },
            }
            for name, value in (("fp32_absmax_round_cast", 200.0), ("two_dynamic_quant_transpose", 100.0))
        }
        self.reports = {
            "public": {"cases": [{"case_name": "public", "status": "passed"}]},
            "extended": {"records": records},
            "repeat": {
                "entry": "run",
                "records": [{"case_name": "public", "iteration": index, "status": "passed"} for index in range(20)],
            },
            "memory": {
                "cases": [
                    {"case_name": "public", "entry": entry, "status": "passed"} for entry in ("run", "run_direct")
                ]
            },
            "performance": {"protocol": protocol},
            "baselines": {
                "protocol": protocol,
                "cases": [
                    {
                        "case_name": "public",
                        "status": "passed",
                        "candidates": candidates,
                        "selected": "two_dynamic_quant_transpose",
                        "selected_wall_us_median": 100.0,
                    }
                ],
            },
        }
        for report in self.reports.values():
            report.update(status="passed", npu_executed=True)

    def _record(self, name: str, entry: str) -> dict[str, str]:
        return {"check": name, "entry": entry, "status": "passed"}

    def _check(self) -> None:
        _REPORTS._validate_reports(self.reports, self.coverage)

    def test_complete_evidence_is_accepted(self) -> None:
        self._check()

    def test_failed_report_is_rejected(self) -> None:
        for name in self.reports:
            with self.subTest(report=name):
                self.reports[name]["status"] = "failed"
                with self.assertRaises(ValueError):
                    self._check()
                self.reports[name]["status"] = "passed"

    def test_failed_repeat_record_cannot_hide_behind_passed_summary(self) -> None:
        self.reports["repeat"]["records"][-1]["status"] = "failed"
        with self.assertRaises(ValueError):
            self._check()

    def test_missing_repeat_file_is_rejected(self) -> None:
        del self.reports["repeat"]
        with self.assertRaises(KeyError):
            self._check()

    def test_duplicate_repeats_do_not_supply_missing_iterations(self) -> None:
        self.reports["repeat"]["records"][-1] = copy.deepcopy(self.reports["repeat"]["records"][0])
        with self.assertRaises(ValueError):
            self._check()

    def test_no_qualified_baseline_is_rejected(self) -> None:
        for candidate in self.reports["baselines"]["cases"][0]["candidates"].values():
            candidate.update(qualified=False, exclusion_reason="precision failed")
        with self.assertRaises(ValueError):
            self._check()

    def test_one_qualified_baseline_is_sufficient(self) -> None:
        self.reports["baselines"]["cases"][0]["candidates"]["fp32_absmax_round_cast"].update(
            qualified=False, exclusion_reason="precision failed"
        )
        self._check()

    def test_slower_selected_baseline_is_rejected(self) -> None:
        self.reports["baselines"]["cases"][0].update(selected="fp32_absmax_round_cast", selected_wall_us_median=200.0)
        with self.assertRaises(ValueError):
            self._check()

    def test_invalid_baseline_timing_is_rejected(self) -> None:
        timing = self.reports["baselines"]["cases"][0]["candidates"]["two_dynamic_quant_transpose"]["timing"]
        for values in ([100.0] * 4, [float("nan")] * 5, [-1.0] * 5):
            with self.subTest(values=values):
                timing["wall_us_rounds"] = values
                with self.assertRaises(ValueError):
                    self._check()

    def test_missing_engineering_or_memory_record_is_rejected(self) -> None:
        for name, field in (("extended", "records"), ("memory", "cases")):
            with self.subTest(report=name):
                record = self.reports[name][field].pop()
                with self.assertRaises(ValueError):
                    self._check()
                self.reports[name][field].append(record)

    def test_non_npu_evidence_is_rejected(self) -> None:
        self.reports["performance"]["npu_executed"] = False
        with self.assertRaises(ValueError):
            self._check()

    def test_mismatched_timing_protocol_is_rejected(self) -> None:
        self.reports["baselines"]["protocol"] = {"warmup": 1, "iterations": 1, "rounds": 1}
        with self.assertRaises(ValueError):
            self._check()


if __name__ == "__main__":
    unittest.main()
