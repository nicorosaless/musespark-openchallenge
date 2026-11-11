"""Compile the layer's own Triton kernels for sm_90 without a GPU, in the judge's image.

    singularity exec -B <repo root>:/w /data/okc/sglang-a55ab7d.sif \
        python3 /w/kernels/muse_decode_layer/tools/compile_sm90.py /w/kernels/muse_decode_layer/submission.py

SGLang's modules are stubbed, so importing the submission touches no CUDA, except the Triton
module whose kernel the layer launches directly (it imports without a GPU). Each kernel written
for the layer is compiled in every specialization kernel_fn launches (the token-count ranges of the
scored shapes and tests). A Triton front-end error (a name redefined across a branch, a shape
mismatch) shows up here in seconds instead of costing an okc check. Kernels copied unchanged
from the single-op problems are compiled by their own checks and are not repeated here.
"""

import importlib.util
import sys
import types

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


class _Stub(types.ModuleType):
    """A module whose every attribute is a dummy callable, for `from x import y`."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


def _stub(*names):
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), _Stub(".".join(parts[:i])))


# The real module first, so its parent packages are real and the stubs only fill in the rest.
import sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton  # noqa: E402,F401

_stub("sgl_kernel.flash_attn", "sgl_kernel.scalar_type",
      "sglang.kernels.ops.activation.activation", "sglang.kernels.ops.kvcache.kvcache",
      "sglang.kernels.ops.moe.moe_wna16_marlin",
      "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing",
      "sglang.kernels.ops.muse_spark_v12.fused_norm_gate",
      "sglang.kernels.ops.muse_spark_v12.fused_norm_residual",
      "sglang.kernels.ops.muse_spark_v12.fused_qk_norm",
      "sglang.srt.layers.moe.fused_moe_triton", "sglang.srt.layers.muse_spark_layers",
      "sglang.srt.layers.quantization.marlin_utils", "sglang.srt.layers.quantization.marlin_utils_fp4",
      "sglang.srt.layers.quantization.unquant")

spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

TARGET = GPUTarget("cuda", 90, 32)


def compile_one(fn, signature, constexprs, num_warps):
    """signature: {arg: type string}; constexpr arguments are listed in constexprs only."""
    sig = {name: signature.get(name, "constexpr") for name in fn.arg_names}
    src = ASTSource(fn=fn, signature=sig, constexprs=constexprs)
    triton.compile(src, target=TARGET, options={"num_warps": num_warps})


def checks():
    """(label, fn, signature, constexprs, num_warps) for each specialization kernel_fn uses."""
    out = []
    if hasattr(sub, "_proj_kernel_red"):
        for block_m, has_lo in ((16, False), (16, True), (32, False), (32, True), (64, True)):
            out.append((f"_proj_kernel_red BLOCK_M={block_m} HAS_LO={has_lo}", sub._proj_kernel_red,
                        {"x_ptr": "*bf16", "w_ptr": "*bf16", "wlo_ptr": "*bf16", "acc_ptr": "*fp32",
                         "M": "i32", "N": "i32", "K": "i32", "stride_xm": "i32", "SPLIT_K": "i32"},
                        {"BLOCK_N": 64, "BLOCK_M": block_m, "BLOCK_K": 128, "HAS_LO": has_lo}, 4))
        for out_ty in ("*bf16", "*fp32"):
            out.append((f"_splitk_finish out={out_ty}", sub._splitk_finish,
                        {"acc_ptr": "*fp32", "out_ptr": out_ty, "MN": "i32"}, {"BLOCK": 1024}, 4))
    if hasattr(sub, "_c1_gemv"):
        for pipe, warps in ((3, 4), (1, 8)):
            out.append((f"_c1_gemv PIPE={pipe}", sub._c1_gemv,
                        {"x_ptr": "*bf16", "wc_ptr": "*u8", "e_ptr": "*u8", "acc_ptr": "*fp32", "K": "i32",
                         "SPLIT_K": "i32"}, {"PIPE": pipe, "ASM": True, "GDC": True}, warps))
        if hasattr(sub, "_c1_gemv_sk"):
            for pipe, warps in ((3, 4), (1, 8)):
                out.append((f"_c1_gemv_sk PIPE={pipe}", sub._c1_gemv_sk,
                            {"x_ptr": "*bf16", "wc_ptr": "*u8", "e_ptr": "*u8", "acc_ptr": "*fp32", "K": "i32",
                             "TOTAL": "i32", "NUM_CTAS": "i32"}, {"PIPE": pipe, "ASM": True, "GDC": True}, warps))
        if hasattr(sub, "_c1_gemm_sk"):
            for block_m in (16, 32):
                out.append((f"_c1_gemm_sk BLOCK_M={block_m}", sub._c1_gemm_sk,
                            {"x_ptr": "*bf16", "wc_ptr": "*u8", "e_ptr": "*u8", "acc_ptr": "*fp32", "M": "i32",
                             "N": "i32", "K": "i32", "stride_xm": "i32", "TOTAL": "i32", "NUM_CTAS": "i32"},
                            {"BLOCK_M": block_m, "PIPE": 3, "ASM": True, "GDC": True, "DOT_FP32": False}, 4))
        for block_m in ((16, 32) if getattr(sub, "C1_MMA_M_MAX", 16) > 16 else (16,)):
            out.append((f"_c1_gemm BLOCK_M={block_m}", sub._c1_gemm,
                        {"x_ptr": "*bf16", "wc_ptr": "*u8", "e_ptr": "*u8", "acc_ptr": "*fp32", "M": "i32",
                         "N": "i32", "K": "i32", "stride_xm": "i32", "SPLIT_K": "i32"},
                        {"BLOCK_M": block_m, "PIPE": 3, "ASM": True, "GDC": True, "DOT_FP32": False}, 4))
        for block in (128, 512):
            for cap in getattr(sub, "C1_CAPS", (sub.C1_CAP,)):
                out.append((f"_c1_finish BLOCK={block} CAP={cap}", sub._c1_finish,
                            {"acc_ptr": "*fp32", "out_ptr": "*bf16", "corr_ptr": "*i16", "PSTRIDE": "i32",
                             "x_ptr": "*bf16", "M": "i32", "N": "i32", "stride_xm": "i32"},
                            {"CAP": cap, "BLOCK": block, "GDC": True}, 4))
    if hasattr(sub, "_route_pre_expert_kernel"):
        concat = "LOGIT_LO" in sub._route_pre_expert_kernel.arg_names
        # (x dtype, LOGIT_LO): the pooler's BF16 output and its own logits; (v47+) the concatenated
        # pooler + router GEMM's FP32 output, logits = hi + lo.
        for x_ty, lo in (("*bf16", 0),) + ((("*fp32", sub.N_EXPERTS),) if concat else ()):
            sig = {"x_ptr": x_ty, "w_ptr": "*fp32", "logit_ptr": "*fp32", "bias_ptr": "*fp32",
                   "start_ptr": "*i32", "out_ptr": "*bf16", "ids_ptr": "*i32", "wts_ptr": "*fp32",
                   "local_ptr": "*i32", "eps": "fp32", "mult": "fp32", "route_eps": "fp32"}
            const = {"D": sub.D_EXPERT, "NE": sub.N_EXPERTS, "TOPK": sub.TOP_K, "N_LOCAL": sub.LOCAL_EXPERTS}
            if concat:
                sig.update({"stride_x": "i32", "stride_logit": "i32"})
                const["LOGIT_LO"] = lo
            out.append((f"_route_pre_expert_kernel x={x_ty} LOGIT_LO={lo}", sub._route_pre_expert_kernel,
                        sig, const, 2))
    if hasattr(sub, "_post_expert_norm_weighted_sum_kernel"):
        out.append(("_post_expert_norm_weighted_sum_kernel (SGLang)", sub._post_expert_norm_weighted_sum_kernel,
                    {"x_ptr": "*bf16", "w_ptr": "*bf16", "s_ptr": "*fp32", "out_ptr": "*bf16", "ids_ptr": "*i32",
                     "eps": "fp32"},
                    {"TOPK": sub.TOP_K, "H": sub.D_EXPERT, "APPLY_NORM": True, "HAS_WEIGHT": True,
                     "WEIGHT_FIRST": True, "HAS_IDS": True, "BLOCK_H": sub.D_EXPERT}, 8))
    if hasattr(sub, "_ffn_boundary_kernel"):
        fused = "ACC" in sub._ffn_boundary_kernel.arg_names
        # (f dtype, ACC, CORR): the boundary on the BF16 slice, and (v39+) on the unpooler's FP32
        # workspace, with the c1 corrections (one token) or without (split-K, 2-8 tokens).
        caps = getattr(sub, "C1_CAPS", (sub.C1_CAP,))
        for f_ty, acc, corr, cap in ((("*bf16", False, False, caps[0]),)
                                     + tuple(("*fp32", True, True, c) for c in caps if fused)
                                     + ((("*fp32", True, False, caps[0]),) if fused else ())):
            sig = {"f_ptr": f_ty, "s_ptr": "*fp32", "hid_ptr": "*bf16", "s_out_ptr": "*fp32",
                   "post_eps": "fp32", "pre_eps": "fp32", "temperature": "fp32", "gate_eps": "fp32",
                   # BF16 weights with gate_ptr (v32 on), FP32 copies with alpha_ptr / beta_ptr before
                   **({"post_w_ptr": "*bf16", "gate_ptr": "*bf16", "pre_w_ptr": "*bf16"}
                      if "gate_ptr" in sub._ffn_boundary_kernel.arg_names else
                      {"post_w_ptr": "*fp32", "alpha_ptr": "*fp32", "beta_ptr": "*fp32", "pre_w_ptr": "*fp32"})}
            const = {"H": sub.HIDDEN, "C": sub.UNPOOL_SLICE}
            if fused:
                sig.update({"corr_ptr": "*i16" if corr else "*fp32", "PSTRIDE": "i32",
                            "x_ptr": "*bf16" if corr else "*fp32", "stride_xm": "i32"})
                const.update({"ACC": acc, "CORR": corr, "CAP": cap})
            out.append((f"_ffn_boundary_kernel C=1024 f={f_ty} ACC={acc} CORR={corr} CAP={cap}",
                        sub._ffn_boundary_kernel,
                        sig, const, 16))
    return out


def cuda_checks():
    """(label, constants) for the attention's CUDA kernel A in each configuration the layer can
    send it (TUNING marks which tiles run in CUDA), compiled with NVRTC for sm_90a. Without a
    GPU _cufa_compile stops at the cubin."""
    if not hasattr(sub, "_cufa_compile"):
        return []
    out = []
    for (layer, tile), tune in sub.TUNING.items():
        if not tune.get("cuda"):
            continue
        for page in (1, 64):
            cfg = {"num_q_heads": 16, "num_kv_heads": 2, "q_len": 1 if tile == "decode" else 8, "page_size": page,
                   "window": 0 if layer == "full" else 2048, "rope_theta": 0.0 if layer == "full" else 500000.0}
            kw = {"cl": tune.get("cl", 0)} if "cl" in sub._cufa_consts.__code__.co_varnames else {}
            out.append((f"CUDA kernel A {layer}/{tile} page {page}", sub._cufa_consts(
                cfg, wg=tune["cuda"] if tune["cuda"] in ("wg", "ws", "cl") else False, **kw)))
    return out


if __name__ == "__main__":
    failed = 0
    for label, fn, sig, cx, warps in checks():
        try:
            compile_one(fn, sig, cx, warps)
            print(f"ok    {label}")
        except Exception as e:  # noqa: BLE001 - report every kernel, then fail
            failed += 1
            print(f"FAIL {label}\n{e}")
    for label, k in cuda_checks():
        try:
            sub._cufa_compile(k, "sm_90a")
            print(f"ok    {label}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {label}\n{e}")
    sys.exit(1 if failed else 0)
