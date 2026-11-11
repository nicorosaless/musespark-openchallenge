"""Swap the experts section of a decode_layer submission for a muse_moe_w4a16 submission file.

    python3 patch_moe.py <submission.py> <moe source.py> "<header first line>"   (edits in place)

Copies the module body between its version print and prepare() (kernels and constants; the layer
defines POST_NORM_EPS once; a helper that calls the problem's kernel_fn(data, state), such as a
warm-up, calls moe_experts instead), ports prepare()'s body after its docstring into moe_prepare()
(with the layer's weight name) and its _split_k block and kernel_fn body into the layer's _split_k
and moe_experts (local_start 0, LOCAL_EXPERTS local experts, ids and weights passed in). Check the
result by diff (integrate_moe.sh does).
"""
import re
import sys

p, src, title = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
m = open(src).read().split("\n")

a = next(i for i, l in enumerate(m) if l.startswith('print(f"muse_moe_w4a16')) + 1
while not m[a].strip():
    a += 1
b = next(i for i, l in enumerate(m) if l.startswith("def prepare(weights: Weights):"))
body_mod = "\n".join(m[a:b]).rstrip().replace("POST_NORM_EPS = 1e-8\n", "", 1)
body_mod = body_mod.replace("        kernel_fn(data, state)\n",
                            '        moe_experts(data["hidden_states"], data["topk_ids"], data["topk_weights"], state)\n')
assert "kernel_fn(" not in body_mod, "the module body calls kernel_fn some other way: port it by hand"


def after_docstring(text, start):
    """Index just past the docstring of the function whose def starts at start."""
    q = text.index('"""', start)
    return text.index('"""', q + 3) + 3


# prepare()'s body after its docstring replaces moe_prepare()'s.
src_text = "\n".join(m)
pb = src_text.index("def prepare(weights: Weights):")
pe = src_text.index("\n\n\n", pb)
prep_body = src_text[after_docstring(src_text, pb):pe].replace('weights["post_norm_w"]', 'weights["post_expert_norm"]')
mb = s.index("def moe_prepare(weights):")
me = s.index("\n\n\n", mb)
s = s[:after_docstring(s, mb)] + prep_body + s[me:]

sk = next(i for i, l in enumerate(m) if l.startswith("# K slices"))
kf = next(i for i, l in enumerate(m) if l.startswith("def kernel_fn("))
split_k = "\n".join(m[sk:kf]).rstrip()
body = "\n".join(m[kf:]).rstrip()
body = body[body.index('    x = data["hidden_states"]'):]
old = ('    x = data["hidden_states"].contiguous()\n'
       '    ids = data["topk_ids"].to(torch.int32).contiguous()  # no-op for the int32 the problem sends\n'
       '    wts = data["topk_weights"].float().contiguous()')
assert old in body
body = body.replace(old, '    x = x.contiguous()\n'
                         '    ids = ids.to(torch.int32).contiguous()  # no-op for the int32 local ids the layer sends\n'
                         '    wts = wts.float().contiguous()')
body = body.replace('    n_local, start = cfg["local_count"], cfg["local_start"]', "    n_local, start = LOCAL_EXPERTS, 0")
assert not re.search(r"\bcfg\b", body), "kernel_fn still reads the problem's config"

h0 = s.index("# Experts: muse_moe_w4a16's")
h1 = s.index("\n\n", h0) + 2  # the header comment ends at the first blank line
header = s[h0:h1]
header = header.split("\n", 1)
header = "# Experts: " + title + "\n" + "\n".join(header[1].split("\n")[1:])
i = s.index("def moe_prepare(weights):")
s = s[:h0] + header + body_mod + "\n\n\n" + s[i:]

i = s.index("# K slices for 7 <= bs <= 64.")
j = s.index("def moe_experts(x, ids, wts, state):")
s = s[:i] + split_k + "\n\n\n" + s[j:]
i = s.index("def moe_experts(x, ids, wts, state):")
doc_end = s.index('"""', s.index('"""', i) + 3) + 3
j = s.index("# ---------------------------------------------------------------------------------------------", i)
s = s[:doc_end] + "\n" + body + "\n\n\n" + s[j:]
open(p, "w").write(s)
print("patched", p)
