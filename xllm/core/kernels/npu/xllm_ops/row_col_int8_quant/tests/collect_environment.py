"""Capture actual versions, allocator settings and source/build fingerprints."""
import hashlib
import ctypes
import configparser
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import torch
import torch_npu

root = Path(__file__).resolve().parents[1]


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def read(path):
    p = Path(path)
    return p.read_text() if p.exists() else "not present"


def runtime_library(cann):
    path = cann / "lib64/libascendcl.so"
    result = {"libascendcl_realpath": str(path.resolve())}
    try:
        library = ctypes.CDLL(str(path))
        library.aclrtGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int32)] * 3
        library.aclrtGetVersion.restype = ctypes.c_int
        major, minor, patch = (ctypes.c_int32() for _ in range(3))
        result["aclrtGetVersion_returncode"] = library.aclrtGetVersion(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        result["ascendcl_api_version_tuple"] = [major.value, minor.value, patch.value]
        library.aclsysGetVersionStr.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        library.aclsysGetVersionStr.restype = ctypes.c_int
        result["package_versions"] = {}
        for name in ("runtime", "hccl", "driver"):
            output = ctypes.create_string_buffer(4096)
            status = library.aclsysGetVersionStr(ctypes.create_string_buffer(name.encode()), output)
            result["package_versions"][name] = {"returncode": status, "version": output.value.decode()}
    except (AttributeError, OSError) as exc:
        result["query_error"] = repr(exc)
    config = configparser.RawConfigParser(strict=False)
    config.read(cann / "aarch64-linux/data/platform_config/Ascend910_9382.ini")
    result["soc_core_and_ub_config"] = {
        f"{section}.{key}": value for section in config.sections() for key, value in config.items(section)
        if "core" in key.lower() or key == "ub_size"}
    return result


def main():
    torch.npu.set_device(0)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    try:
        import torch_npu.version as npu_version
        npu_git = getattr(npu_version, "git_version", "unknown")
    except ImportError:
        npu_git = "unknown"
    versions = {}
    for name in ("torch", "torch_npu", "numpy", "ml_dtypes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    report = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "os_release": read("/etc/os-release"),
        "versions": versions, "torch_npu_git_version": npu_git,
        "device": torch.npu.get_device_name(0), "logical_device": 0,
        "logical_devices_visible": torch.npu.device_count(),
        "loaded_numpy_version": numpy.__version__,
        "runtime_library": runtime_library(cann),
        "environment": {k: os.environ.get(k, "unset") for k in (
            "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_CUSTOM_OPP_PATH",
            "PYTORCH_NPU_ALLOC_CONF", "TASK_QUEUE_ENABLE")},
        "cann_installation": {str(p): read(p) for p in (
            cann / "version.info", cann / "aarch64-linux/ascend_toolkit_install.info")},
        "driver_version_info": read("/usr/local/Ascend/driver/version.info"),
        "npu_smi": command(["npu-smi", "info"]),
        "source_revision": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "source_worktree": command(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]),
        "acceptance_environment_note": "User-supplied DevEnv; compare actual versions with task requirements."
    }
    manifest = {}
    tracked = command(["git", "-C", str(root), "ls-files", "-z"])
    if tracked["returncode"] == 0 and tracked["stdout"]:
        selected = [root / name for name in tracked["stdout"].split("\0") if name]
    else:
        selected = list(root.glob("*.py")) + list(root.glob("*.sh")) + list(root.glob("*.json")) + list(root.glob("*.md"))
        selected += [root / "CMakeLists.txt"]
        for directory in ("op_host", "op_kernel", "direct", "python", "examples", "tests", "docs"):
            selected += [p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in sorted(set(selected)):
        manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("build_out/custom_opp_ubuntu_aarch64.run", "build_binding/rowcol_ext.so",
                 "build_direct/lib/librowcol_direct.so", "build_examples/aclnn_example"):
        path = root / name
        if path.is_file():
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["implementation_sha256"] = manifest
    output = root / "reports/environment.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
