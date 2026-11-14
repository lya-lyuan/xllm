"""NPU-only public entry points. There is deliberately no CPU compute fallback."""
import importlib.util
from pathlib import Path

import torch
import torch_npu  # noqa: F401

_extension = None


def _backend():
    global _extension
    if _extension is None:
        path = Path(__file__).resolve().parent / "build_binding/rowcol_ext.so"
        spec = importlib.util.spec_from_file_location("rowcol_ext", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _extension = module
    return _extension


run = _backend().run
run_direct = _backend().run_direct


def prepare(inputs, attributes):
    backend = _backend()
    x = backend.prepare_input(inputs, attributes)
    return lambda: backend.run_tensor(x)
