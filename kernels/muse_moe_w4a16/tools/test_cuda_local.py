"""The v3 CUDA GEMMs on the dev box's GPU (sm75, no BF16 MMA): compiled with
-DOKC_EMULATE, which swaps mma.sync / mul.bf16x2 / cvt.bf16 for FP32 + warp shuffles.
Checks launching, arguments, indexing, packing and epilogues against the reference, at
H = I = 512 (K slices of 512), 8 experts of which 4 are local, top 2. Small on purpose:
never run prepare() on the full 256-expert bank on the dev box.

    ‹local› -u test_cuda_local.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submission  # noqa: E402
from checks import compare  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402

submission.CUDA_OPTIONS = ["-std=c++17", "-DOKC_EMULATE"]
H = I = int(os.environ.get("OKC_TEST_DIM", "512"))
N_EXPERTS = int(os.environ.get("OKC_TEST_E", "8"))
LOCAL = int(os.environ.get("OKC_TEST_LOCAL", "4"))
TOP_K = int(os.environ.get("OKC_TEST_TOPK", "2"))
DEV = "cuda"


def weights(seed, tiny_w2_scales=False):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    w13, sf13, gs13 = random_weights(2 * I, H, generator=gen, device=DEV, leading=(N_EXPERTS,))
    w2, sf2, gs2 = random_weights(H, I, generator=gen, device=DEV, leading=(N_EXPERTS,))
    gs2 = gs2.expand(N_EXPERTS).contiguous()
    if tiny_w2_scales:  # E4M3 codes 0..15 (zero, subnormal, small normal): only the exact decode is right
        sf2 = torch.randint(0, 16, sf2.shape, generator=gen, device=DEV, dtype=torch.uint8).view(torch.float8_e4m3fn)
        gs2 = gs2 * 2.0 ** 13
    return {"w13": w13, "w13_sf": sf13, "w13_gs": gs13.expand(N_EXPERTS).contiguous() * torch.linspace(0.5, 2, N_EXPERTS, device=DEV),
            "w2": w2, "w2_sf": sf2, "w2_gs": gs2,
            "post_norm_w": (0.1 * torch.randn(H, generator=gen, device=DEV)).to(torch.bfloat16)}


def inputs(bs, seed, start, hot):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=DEV)
    if hot:
        scores[:, start + 3] += 10
    routed = scores.topk(TOP_K, dim=-1)
    return {"hidden_states": torch.randn((bs, H), generator=gen, device=DEV).to(torch.bfloat16),
            "topk_ids": routed.indices.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": H, "d_expert": I, "n_experts": N_EXPERTS,
                       "local_start": start, "local_count": LOCAL, "top_k": TOP_K, "sharding": "ep8"}}


def reference(data, w):
    x = data["hidden_states"].float()
    out = torch.zeros_like(x)
    ids = data["topk_ids"].long()
    lo = data["config"]["local_start"]
    for e in torch.unique(ids).tolist():
        if not lo <= e < lo + LOCAL:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        y = inter @ dequantize(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T
        y = y * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


class _NoFloatArgs:
    """torch.cuda._compile_kernel passes a Python float as a C float in torch 2.9 and as a
    C double in 2.14; a float kernel argument read 0.0 on the judge. Fail on any."""

    def __init__(self, kern):
        self.kern = kern

    def __call__(self, *a, **kw):
        bad = [x for x in kw.get("args", []) if isinstance(x, float)]
        assert not bad, f"float kernel argument(s) {bad}: pass ints/tensors or bake constants into the source"
        return self.kern(*a, **kw)


def main():
    failed = 0
    for tiny in (False, True):
        w = weights(1, tiny)
        print("weights ok", flush=True)
        state = submission.prepare(w)
        state["cuda"] = {n: _NoFloatArgs(k) for n, k in state["cuda"].items()}
        torch.cuda.synchronize()
        print("prepare ok", flush=True)
        assert state["w2_exact"] == tiny
        for bs, start, hot in [(1, 0, False), (4, 4, False), (6, 0, True), (16, 0, True), (33, 4, True), (80, 0, False),
                               (300, 0, True), (600, 4, False)]:
            data = inputs(bs, 100 + bs, start, hot)
            print(f"bs={bs} running", flush=True)
            out = submission.kernel_fn(data, state)
            torch.cuda.synchronize()
            ok, msg = compare(out, reference(data, w), 1.5e-2)
            print(f"bs={bs:3d} start={start:2d} hot={hot!s:5} tiny_w2={tiny!s:5} {'ok  ' if ok else 'FAIL'} {msg}")
            failed += not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
