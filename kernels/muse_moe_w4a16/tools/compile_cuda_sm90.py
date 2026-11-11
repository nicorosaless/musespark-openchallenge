import sys, os, re, subprocess
from collections import Counter
sys.path.insert(0, sys.argv[1])
import triton
from torch.cuda._utils import _nvrtc_compile
import submission
CUDA_SRC = submission.cuda_source() if hasattr(submission, "cuda_source") else submission.CUDA_SRC
B = os.path.dirname(triton.__file__) + "/backends/nvidia/bin"
for name in ("moe_gemm1", "moe_gemm2", "moe_gemm1_exact"):
    ptx = _nvrtc_compile(CUDA_SRC, name, "90", "", None, ["-std=c++17"])
    ptx = ptx.decode() if isinstance(ptx, bytes) else ptx
    f = f"‹tmp›/okc-moe/{name}.ptx"; open(f, "w").write(ptx)
    info = subprocess.run([B + "/ptxas", "-arch=sm_90", "-v", f, "-o", f[:-4] + ".cubin"], capture_output=True, text=True)
    print(name, " | ".join(l.split(":", 1)[-1].strip() for l in info.stderr.splitlines() if "registers" in l or "spill" in l or "rror" in l))
    sass = subprocess.run([B + "/nvdisasm", "-c", f[:-4] + ".cubin"], capture_output=True, text=True).stdout.splitlines()
    ops = Counter(m.group(1) for m in (re.search(r"\*/\s+(?:@!?U?P\w+ )?([A-Z][A-Z0-9_]*)", x) for x in sass) if m)
    print("   total", sum(ops.values()), ops.most_common(12))
