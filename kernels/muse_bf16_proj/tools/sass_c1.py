"""Compile the c1 kernels for sm_90 without a GPU (Triton 3.5 of the bf16cpu venv) and print
registers, shared memory and the SASS mix per variant. Outputs (cubin, ttgir, sass) in ‹tmp›/c1sass.

  ‹local› sass_c1.py   # the judge's Triton (3.8.0), no torch
  ‹local› sass_c1.py    # Triton 3.5

Pointers and integers get tt.divisibility=16, as the JIT gives them; without it every load
is a byte load and the SASS says nothing about the real kernel.
"""
import glob
import os
import subprocess
import sys

try:
    import torch  # noqa: F401  (c1.py needs it only outside the kernels)
except ImportError:
    import types
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("‹tmp›/c1sass", exist_ok=True)
print("triton", triton.__version__)
import c1
tgt = GPUTarget("cuda", 90, 32)
BIN = glob.glob(sys.prefix + "/lib/python3*/site-packages/triton/backends/nvidia/bin")[0]

def build(fn, sig, consts, warps=4, stages=3, tag="k", nodiv=()):
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(sig) if sig[n] != "constexpr" and n not in nodiv}
    src_cls = ASTSource
    if getattr(fn, "is_gluon", lambda: False)():
        from triton.experimental.gluon._runtime import GluonASTSource as src_cls
    k = triton.compile(src_cls(fn=fn, signature=sig, constexprs=consts, attrs=attrs), target=tgt,
                       options={"num_warps": warps, "num_stages": stages})
    cub = f"‹tmp›/c1sass/{tag}.cubin"
    open(cub, "wb").write(k.asm["cubin"])
    open(cub[:-6] + ".ttgir", "w").write(k.asm["ttgir"])
    ru = subprocess.run([BIN + "/cuobjdump", "--dump-resource-usage", cub], capture_output=True, text=True).stdout
    reg = ru.split("REG:")[1].split()[0]
    sass = subprocess.run([BIN + "/cuobjdump", "-sass", cub], capture_output=True, text=True).stdout
    open(cub[:-6] + ".sass", "w").write(sass)
    ins = [l.split("*/")[1].split()[0] if not l.split("*/")[1].split()[0].startswith("@") else l.split("*/")[1].split()[1]
           for l in sass.splitlines() if l.strip().startswith("/*") and "*/" in l and l.strip()[2:6].isalnum() and len(l.split("*/")[1].split()) > 0]
    cnt = lambda p: sum(1 for i in ins if i.startswith(p))
    print(f"{tag:14s} regs {reg:>3} smem {k.metadata.shared:6d} instr {len(ins):4d} LDGSTS {cnt('LDGSTS'):2d} "
          f"LDG {cnt('LDG'):2d} LDS {cnt('LDS'):2d} STS {cnt('STS'):2d} BAR {cnt('BAR'):2d} PRMT {cnt('PRMT'):3d} "
          f"FFMA {cnt('FFMA'):2d} FMUL {cnt('FMUL'):2d} FADD {cnt('FADD'):2d}")
    return k

base = {"x_ptr": "*bf16", "wc_ptr": "*u8", "e_ptr": "*u8", "acc_ptr": "*fp32", "N": "i32", "K": "i32", "SPLIT_K": "i32"}
if __name__ == "__main__":
    sig = dict(base, ASM="constexpr", GDC="constexpr", UNROLL="constexpr", DECODE="constexpr", PREFETCH="constexpr",
               PIPE="constexpr", corr_ptr="*i16", PSTRIDE="i32", CORR="constexpr", CAP="constexpr")
    for w, p, corr in ((4, 1, False), (4, 3, False), (8, 1, False), (8, 3, False), (4, 1, True), (4, 3, True),
                       (8, 1, True)):
        build(c1._gemv_c1, sig, dict(ASM=True, GDC=True, UNROLL=1, DECODE=True, PREFETCH=False, PIPE=p, CORR=corr,
                                     CAP=c1.C1_CAP), warps=w, tag=f"v1w{w}p{p}{'corr' if corr else ''}",
              nodiv=("SPLIT_K",))
    sigf = {"acc_ptr": "*fp32", "out_ptr": "*bf16", "corr_ptr": "*i16", "PSTRIDE": "i32", "x_ptr": "*bf16", "N": "i32",
            "CAP": "constexpr", "BLOCK": "constexpr", "GDC": "constexpr", "HAS_CORR": "constexpr", "EARLY": "constexpr"}
    for hc, early in ((True, False), (True, True), (False, False)):
        build(c1._finish_c1, sigf, dict(CAP=c1.C1_CAP, BLOCK=128, GDC=True, HAS_CORR=hc, EARLY=early), warps=4,
              tag=f"fin{'C' if hc else ''}{'E' if early else ''}")
    sigw = dict(base, wc_ptr="*i32", ASM="constexpr", GDC="constexpr", PIPE="constexpr")
    for w in (4, 8):
        for p in (1, 2, 3, 4):
            build(c1._gemv_c1w, sigw, dict(ASM=True, GDC=True, PIPE=p), warps=w, tag=f"v2w{w}p{p}", nodiv=("SPLIT_K",))
