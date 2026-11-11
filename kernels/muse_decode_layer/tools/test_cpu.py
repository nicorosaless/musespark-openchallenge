"""The layer's own Triton kernels against plain PyTorch, on the CPU (Triton's interpreter).

    singularity exec -B <repo root>:/w /data/okc/sglang-a55ab7d.sif \
        python3 /w/kernels/muse_decode_layer/tools/test_cpu.py /w/kernels/muse_decode_layer/submission.py

Checks values, not speed: the routing fused with the pre-expert norm and the local ids, and the
post-FFN boundary read from the unpooler's slice.
"""

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import compile_sm90  # noqa: E402,F401  (stubs SGLang and loads the submission)

sub = compile_sm90.sub
torch.manual_seed(0)
failed = 0


def close(label, got, want, tol):
    global failed
    err = float((got.float() - want.float()).norm() / want.float().norm().clamp(min=1e-12))
    ok = err <= tol
    failed += not ok
    print(f"{'ok   ' if ok else 'FAIL '} {label}: relative error {err:.2e} (tol {tol:g})")


# c1 (lossless compressed BF16): round trip bit for bit on the layer's shapes, then the one-token
# GEMV in the interpreter (plain-Triton decode, no PDL), against x @ W.T.
if hasattr(sub, "compress_c1"):
    sub.C1_ASM, sub.C1_GDC = False, False
    for name, (n, k) in (("o", (8192, 1024)), ("unpooler_slice", (1024, 4096)), ("qkv_gate", (2304, 8192))):
        w = (torch.randn(n, k) / k ** 0.5).to(torch.bfloat16)
        w[0, :5] = 0  # zeros and tiny values take the correction table
        w[1, 7] = 1e-9
        for cap in getattr(sub, "C1_CAPS", (sub.C1_CAP,)):
            comp = sub.compress_c1(w, cap) if hasattr(sub, "C1_CAPS") else sub.compress_c1(w)
            back = sub.decompress_c1(comp, n, k) if comp is not None else None
            ok = back is not None and torch.equal(back.view(torch.int16), w.view(torch.int16))
            failed += not ok
            print(f"{'ok   ' if ok else 'FAIL '} c1 round trip {name} cap {cap}")
        if ok and name in ("unpooler_slice", "qkv_gate"):  # qkv_gate: the stream-K GEMV (v11b)
            x = torch.randn(1, k).to(torch.bfloat16)
            ws = {"acc": torch.zeros(n), "num_sms": 132}
            got = sub._c1_project(x, comp, n, ws)
            close(f"c1 GEMV {name} m=1", got, (x.float() @ w.float().T).to(torch.bfloat16), 1e-2)
            ok = not ws["acc"].any()
            failed += not ok
            print(f"{'ok   ' if ok else 'FAIL '} c1 workspace left at zero")

# c1 on tensor cores (_c1_gemm + _c1_finish), 2-32 tokens, on a small weight; FP32 dots (the
# interpreter's BF16 tl.dot is wrong).
if hasattr(sub, "compress_c1"):
    sub.C1_ASM, sub.C1_GDC, sub.C1_DOT_FP32 = False, False, True
    n, k = 256, 1024
    w = (torch.randn(n, k) / k ** 0.5).to(torch.bfloat16)
    w[2, :3] = 0
    comp = sub.compress_c1(w)
    # 132 SMs: 4 tiles x split 8 leave SMs short, so stream-K if the submission has it (_c1_gemm_sk,
    # v43); 4 SMs: K split. Up to C1_MMA_M_MAX tokens (16 or 32).
    for m, sms in ((9, 132), (sub.C1_MMA_M_MAX, 132), (9, 4), (sub.C1_MMA_M_MAX, 4)):
        x = torch.randn(m, k).to(torch.bfloat16)
        ws = {"acc": torch.zeros(m * n), "num_sms": sms}
        got = sub._c1_project(x, comp, n, ws)
        close(f"c1 GEMM m={m} num_sms={sms}", got, (x.float() @ w.float().T).to(torch.bfloat16), 1e-2)
        ok = not ws["acc"].any()
        failed += not ok
        print(f"{'ok   ' if ok else 'FAIL '} c1 GEMM m={m} num_sms={sms} workspace left at zero")

# Routing + pre-expert norm + local ids (route_pre_expert), against PyTorch and the routing's math.
for tokens in (1, 5):
    x = torch.randn(tokens, 4096).to(torch.bfloat16)
    logits = 8 * torch.randn(tokens, 256)
    bias = 0.055 * torch.randn(256)
    gain = 0.1 * torch.randn(4096)
    out, wts, ids, local = sub.route_pre_expert(x, logits, torch.tensor([224], dtype=torch.int32),
                                                {"pre_expert_w": gain + 1.0, "router_bias": bias})
    score = torch.sigmoid(logits * sub.ROUTER_MULT)
    want_ids = (score + bias).topk(8, dim=-1).indices
    ok = torch.equal(ids.long(), want_ids)
    failed += not ok
    print(f"{'ok   ' if ok else 'FAIL '} route_pre_expert top-8 T={tokens}")
    picked = score.gather(1, want_ids)
    close(f"route_pre_expert weights T={tokens}", wts, picked / (picked.sum(-1, keepdim=True) + 1e-15), 1e-6)
    xf = x.float()
    want = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * (1 + gain)).to(torch.bfloat16)
    close(f"route_pre_expert norm T={tokens}", out, want, 1e-2)
    want_local = (want_ids - 224) % 256
    ok = torch.equal(local.long(), torch.where(want_local < 32, want_local, -1))
    failed += not ok
    print(f"{'ok   ' if ok else 'FAIL '} route_pre_expert local ids T={tokens}")

# Routing from the concatenated pooler + router GEMM (v47): FP32 pooler columns and the logits as
# hi + lo halves, column slices of one [T, 4096 + 512] tensor; against the plain call.
if "logit_lo" in sub.route_pre_expert.__code__.co_varnames:
    tokens = 5
    pr = torch.randn(tokens, 4096 + 512)
    pr[:, 4096:4352] *= 8
    pr[:, 4352:] *= 1e-3
    bias = 0.055 * torch.randn(256)
    gain = 0.1 * torch.randn(4096)
    st = {"pre_expert_w": gain + 1.0, "router_bias": bias}
    start = torch.tensor([224], dtype=torch.int32)
    got = sub.route_pre_expert(pr[:, :4096], pr[:, 4096:], start, st, 256)
    want = sub.route_pre_expert(pr[:, :4096].contiguous(), pr[:, 4096:4352] + pr[:, 4352:], start, st)
    ok = all(torch.equal(g, w_) for g, w_ in zip(got[2:], want[2:]))
    failed += not ok
    print(f"{'ok   ' if ok else 'FAIL '} route_pre_expert concatenated: same ids")
    close("route_pre_expert concatenated weights", got[1], want[1], 1e-6)
    close("route_pre_expert concatenated norm", got[0], want[0], 1e-6)

# Post-FFN boundary from the slice, against the reference's ref_back tail.
for tokens in (1, 8):
    unpooled = torch.randn(tokens, 1024).to(torch.bfloat16)
    stream = torch.randn(tokens, 8192)
    post, pre, gate = 0.1 * torch.randn(8192), 0.1 * torch.randn(8192), 0.3 * torch.randn(8192)
    coef = sub._residual_coefficients(gate)
    state = {"ffn_boundary_w": {"post": post + 1, "pre": pre + 1, **coef}}
    hidden, new = sub.ffn_boundary_sliced(unpooled, stream, state)
    f = torch.zeros(tokens, 8192)
    f[:, :1024] = unpooled.float()
    f = f * (1 + post)
    rms = lambda v, eps: v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)  # noqa: E731
    want_s = coef["alpha"] * stream + coef["beta"] * rms(f, 1e-8)
    close(f"ffn_boundary stream T={tokens}", new, want_s, 1e-6)
    close(f"ffn_boundary hidden T={tokens}", hidden, (rms(want_s, 1e-5) * (1 + pre)).to(torch.bfloat16), 1e-2)

# The unpooler's finish folded into the boundary (unpool_ffn_boundary), against the two-launch path
# (_dense + ffn_boundary_sliced): one token through c1 (corrections included), and the boundary
# straight from a filled FP32 workspace (the split-K path's 2-8 tokens).
if hasattr(sub, "unpool_ffn_boundary"):
    sub.C1_ASM, sub.C1_GDC = False, False
    n, k = sub.UNPOOL_SLICE, 4096
    w = (torch.randn(n, k) / k ** 0.5).to(torch.bfloat16)
    w[0, :5] = 0
    w[3, 9] = 1e-9
    comp = sub.compress_c1(w, 8) if hasattr(sub, "C1_CAPS") else sub.compress_c1(w)  # v44: the layer's cap
    post, pre, gate = 0.1 * torch.randn(8192), 0.1 * torch.randn(8192), 0.3 * torch.randn(8192)
    coef = sub._residual_coefficients(gate)
    state = {"ffn_boundary_w": {"post": post + 1, "pre": pre + 1, **coef}, "c1": {"unpooler_slice": comp},
             "dense": {"unpooler_slice": w}, "packed": {},
             "ws": {"unpooler_slice": {"acc": torch.zeros(8 * n), "num_sms": 132}}}
    y = torch.randn(1, k).to(torch.bfloat16)
    stream = torch.randn(1, 8192)
    want_h, want_s = sub.ffn_boundary_sliced(sub._dense(y, "unpooler_slice", state), stream, state)
    got_h, got_s = sub.unpool_ffn_boundary(y, stream, state)
    close("unpool_ffn_boundary c1 stream T=1", got_s, want_s, 1e-6)
    close("unpool_ffn_boundary c1 hidden T=1", got_h, want_h, 1e-2)
    ok = not state["ws"]["unpooler_slice"]["acc"].any()
    failed += not ok
    print(f"{'ok   ' if ok else 'FAIL '} unpool_ffn_boundary leaves the workspace at zero")
    tokens = 5
    acc = torch.randn(tokens, n)
    stream = torch.randn(tokens, 8192)
    # Triton's interpreter rounds FP32 -> BF16 toward zero (the GPU, cvt.rn, to nearest even, as
    # _splitk_finish does there), so the reference truncates too.
    unpooled = (acc.view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)
    want_h, want_s = sub.ffn_boundary_sliced(unpooled, stream, state)
    ws = acc.reshape(-1).clone()
    got_h, got_s = sub._ffn_boundary_launch(ws, stream, state, tokens, n, acc=True)
    close(f"boundary from the workspace stream T={tokens}", got_s, want_s, 1e-6)
    close(f"boundary from the workspace hidden T={tokens}", got_h, want_h, 1e-2)
    ok = not ws.any()
    failed += not ok
    print(f"{'ok   ' if ok else 'FAIL '} boundary from the workspace leaves it at zero")

sys.exit(1 if failed else 0)
