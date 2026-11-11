"""Bit layouts for the E2M1 decode where registers 2 and 3 take one IMAD instead of two
shifts: r = ((q & SRC) * (2^a + 2^b)) & 0x81C081C0, with the sign from the 2^a copy and the
magnitude from the 2^b copy. Register 0 is q & 0x81C081C0 (no shift), register 1 is
(q << k) & 0x81C081C0. Per 16-bit half, a code is (sign bit, 3 magnitude bits); the target is
sign at 15 and magnitude at 8..6 of each half.

A layout is valid when, for registers 2 and 3:
  - the two shifted copies of (q & SRC) never set the same bit (no carries, so the product is
    their OR), for every q;
  - the bits of each copy that are not the wanted ones land outside the target positions
    (or above bit 31);
  - the four codes of a half use 16 distinct positions.
Prints every valid (k, a2, b2, a3, b3) with the CODE_SIGN_BIT / CODE_MAG_LOW it implies.
"""
from itertools import product

TARGET = {15, 8, 7, 6}
T32 = TARGET | {p + 16 for p in TARGET}


def code_bits(sign, mag_low):
    return {sign, mag_low, mag_low + 1, mag_low + 2}


def imad_ok(sign_src, mag_low_src, a, b):
    src = code_bits(sign_src, mag_low_src)
    src32 = src | {p + 16 for p in src}
    A = {p + a for p in src32}
    B = {p + b for p in src32}
    if A & B:
        return False  # overlapping copies: carries
    want_a = {sign_src + a, sign_src + 16 + a}
    want_b = {p + b for p in (mag_low_src, mag_low_src + 1, mag_low_src + 2)}
    want_b |= {p + 16 for p in want_b}
    if want_a != {15, 31} or want_b != {6, 7, 8, 22, 23, 24}:
        return False
    stray = {p for p in (A - want_a) | (B - want_b) if p < 32}
    return not (stray & T32)


found = []
for k in range(1, 16):
    c1 = (15 - k, 6 - k)
    if c1[1] < 0:
        continue
    for a2, b2, a3, b3 in product(range(1, 16), range(1, 7), range(1, 16), range(1, 7)):
        c2 = (15 - a2, 6 - b2)
        c3 = (15 - a3, 6 - b3)
        codes = [(15, 6), c1, c2, c3]
        bits = [code_bits(*cd) for cd in codes]
        if any(min(bs) < 0 or max(bs) > 15 for bs in bits):
            continue
        if len(set().union(*bits)) != 16:
            continue
        if not (imad_ok(*c2, a2, b2) and imad_ok(*c3, a3, b3)):
            continue
        found.append((k, a2, b2, a3, b3, tuple(cd[0] for cd in codes), tuple(cd[1] for cd in codes)))

for f in found:
    k, a2, b2, a3, b3, sb, ml = f
    print(f"k={k} r2: a={a2} b={b2} r3: a={a3} b={b3}  CODE_SIGN_BIT={sb} CODE_MAG_LOW={ml}")
print(len(found), "layouts")
