"""Diagnostic trace only; the public benchmark remains the acceptance timer."""
import json
from pathlib import Path
import sys

import torch
import torch_npu

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import solution

x = torch.randn(257, 1026, device="npu:0", dtype=torch.float16)
for _ in range(20):
    solution.run([x], {})
torch.npu.synchronize()
path = root / "logs/device_trace.json"
with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                          torch_npu.profiler.ProfilerActivity.NPU]) as profiler:
    for _ in range(10):
        solution.run([x], {})
    torch.npu.synchronize()
profiler.export_chrome_trace(str(path))
trace = json.loads(path.read_text())
events = trace["traceEvents"] if isinstance(trace, dict) else trace
for event in events:
    if "RowColInt8" in event.get("name", ""):
        print(json.dumps(event), flush=True)
