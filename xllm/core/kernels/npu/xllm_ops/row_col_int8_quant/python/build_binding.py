"""Build the binding once; solution imports the resulting extension without JIT."""
import os
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0"))
npu = Path(torch_npu.__file__).parent
build = root / "build_binding"
build.mkdir(exist_ok=True)
load(
    name="rowcol_ext",
    sources=[str(root / "python/binding.cpp")],
    extra_include_paths=[str(cann / "include"), str(npu / "include")],
    extra_cflags=["-O2", "-g0", f'-DROWCOL_DIRECT_LIBRARY=\\"{root}/build_direct/lib/librowcol_direct.so\\"'],
    extra_ldflags=[f"-L{cann}/lib64", "-lascendcl", "-lnnopbase", f"-L{npu}/lib",
                   "-ltorch_npu", "-ldl"],
    build_directory=str(build),
    verbose=True,
)
