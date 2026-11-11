"""Correctness of submission.py's Triton path on CPU, with TRITON_INTERPRET=1.

Checks the split-K reduction logic (tiling, atomics, last-CTA epilogue, workspace left at
zero) against reference.py. No timing: the interpreter runs programs one after another.

The interpreter's tl.dot is wrong on BF16 operands (errors ~1e11; FP32 is fine), so the
kernel gets FP32 copies of the same BF16 values. That tests the logic, not the precision
of BF16 MMAs: okc check does that.

  TRITON_INTERPRET=1 ‹local› test_cpu.py   (Triton 3.5)
"""

import os
import sys

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

torch.cuda.get_device_properties = lambda _d: type("P", (), {"multi_processor_count": 132})()
torch.cuda.synchronize = lambda *a, **k: None

import numpy as np
from triton.runtime import interpreter

# Triton 3.1's interpreter turns 1-element arrays into Python ints with int(), which
# numpy 2 rejects; range(kb_start, kb_end) in the kernel needs it. Newer Triton is fine.
if hasattr(interpreter, "_patch_lang_tensor"):
    _patch_tensor = interpreter._patch_lang_tensor

    def _patch_lang_tensor(tensor):
        _patch_tensor(tensor)
        tensor.__index__ = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])

    interpreter._patch_lang_tensor = _patch_lang_tensor

import reference
import submission

_project = submission._project


def _project_fp32(x, w, wlo, *args):
    return _project(x.float(), w.float(), None if wlo is None else wlo.float(), *args)


submission._project = _project_fp32
# griddepcontrol (programmatic dependent launch) is inline PTX the interpreter cannot run:
# this test covers the counter path; the PDL path is checked on the GPU (bench, okc).
submission.SPLITK_PDL = False

CASES = [  # (m, layer, dist): TASK.md tests on the Triton path, then scored shapes and edges
    (1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"),
    (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"),
    (1, "router", "normal"), (16, "unpooler", "normal"), (32, "qkv_gate", "outliers"),
    (64, "o", "normal"), (128, "router", "normal"), (256, "router", "outliers"),
    (8, "unpooler", "normal"), (1, "o", "normal"),
    (130, "unpooler", "normal"), (40, "o", "normal"), (40, "router", "normal"), (200, "router", "outliers"),
]


def main():
    ok_all = True  # the CUDA path's compression is checked by a separate GEMV test (not published)
    weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cpu")
    state = submission.prepare(weights)
    assert not state["acc"].any() and not state["cnt"].any()
    for i, (m, layer, dist) in enumerate(CASES):
        for rep in range(2):  # the second call checks the workspace came back clean
            data = reference.generate_input(m, layer, seed=100 + i + rep, dist=dist, device="cpu")
            out = submission.kernel_fn(data, state)
            ok, msg = reference.check_implementation(data, out, weights)
            clean = not state["acc"].any() and not state["cnt"].any()
            ok_all &= ok and clean
            print(f"m={m:4d} {layer:9s} {dist:8s} rep{rep} {'OK ' if ok and clean else 'FAIL'} "
                  f"{msg}{'' if clean else ' (workspace not zeroed)'}", flush=True)
    print("ALL OK" if ok_all else "FAILURES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
