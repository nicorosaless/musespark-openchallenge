"""Swap the attention section of a decode_layer submission for a muse_attention_decode
submission file.

    python3 patch_attn.py <submission.py> <attention source.py> "<header lines>"   (edits in place)

Copies the attention file from RMS_EPS up to kernel_fn, plus every module-level statement after
kernel_fn (the TUNING switches), drops the lazy RoPE table (prepare() builds SGLang's) and the
in-function `from cuda.bindings import` (the layer imports driver and nvrtc once at load time),
and ports kernel_fn to attention(qkv, data, cos_sin).
"""
import re
import sys

p, src, header = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
a = open(src).read()
al = a.split("\n")

start = al.index("RMS_EPS = 1e-5")
kfn = next(i for i, l in enumerate(al) if l.startswith("def kernel_fn"))
kern = "\n".join(al[start:kfn]).rstrip()
i = kern.index("ROPE_POSITIONS = 1 << 21")
j = kern.index("def _regs(cap):")
kern = kern[:i] + kern[j:]
kern = kern.replace("_ROPE = {}\n", "")
n_imports = len(re.findall(r"^ *from cuda\.bindings import [a-z, ]+\n", kern, flags=re.M))
kern = re.sub(r"^ *from cuda\.bindings import [a-z, ]+\n", "", kern, flags=re.M)

# kernel_fn ends where module-level code starts again.
end = next(i for i in range(kfn + 1, len(al)) if al[i] and not al[i][0].isspace() and not al[i].startswith(")"))
body = "\n".join(al[kfn:end]).rstrip()
tail = "\n".join(al[end:]).strip()
assert tail.startswith("TUNING["), tail[:80]
kern += ("\n\n\n# The submitted file sets these switches after kernel_fn; here they close the attention section.\n"
         + tail)

old_hdr = re.search(r"# Attention: muse_attention_decode's .*?\n\n", s, flags=re.S)
att_start = s.index("\ndef attention(qkv, data, cos_sin):")
s = s[:old_hdr.start()] + header.rstrip("\n") + "\n\n" + kern + "\n\n" + s[att_start:]

body = body.replace("def kernel_fn(data: Inputs) -> torch.Tensor:", '''def attention(qkv, data, cos_sin):
    """Gated attention output [T, Hq * 64] BF16 from the fused projection qkv [T, 2304]; writes
    the step's keys and values to the FP8 cache. muse_attention_decode's kernel_fn, with qkv
    passed in (its CUDA launchers read it from data) and the RoPE table built once in prepare()
    (the same [2^21, 64] cos | sin table)."""''')
body = body.replace('    qkv = data["qkv"]\n', '    data = {**data, "qkv": qkv}  # [T, 2304]: the same memory as its [B, L, 2304]\n')
old_rope = ('    rope_table = _rope_table(cfg["rope_theta"], d, page_table.shape[1] * page, device) '
            'if rope else data["k_scale"]')
assert old_rope in body
body = body.replace(old_rope, '''    assert not rope or page_table.shape[1] * page <= ROPE_POSITIONS, "RoPE table too short for this cache"
    rope_table = cos_sin if rope else data["k_scale"]''')
old_out = '    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)'
assert old_out in body
body = body.replace(old_out, '    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)')

i = s.index("def attention(qkv, data, cos_sin):")
j = s.index("# ---------------------------------------------------------------------------------------------", i)
s = s[:i] + body + "\n\n\n" + s[j:]
open(p, "w").write(s)
print("patched", p, "- in-function cuda.bindings imports removed:", n_imports)
