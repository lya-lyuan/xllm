"""Diagnostic only: captured replay removes most host gaps; NOT acceptance timing."""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solution

torch.npu.set_device(0)
for m, n in ((257, 1026), (128, 4096), (1024, 4096), (4096, 4096), (4096, 256)):
    x = torch.randn((m, n), device="npu:0", dtype=torch.float16)
    for _ in range(5):
        solution.run([x], {})
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        ys = solution.run([x], {})
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(200):
        graph.replay()
    end.record()
    end.synchronize()
    print(m, n, "diagnostic_replay_us", start.elapsed_time(end) * 1000 / 200, flush=True)
