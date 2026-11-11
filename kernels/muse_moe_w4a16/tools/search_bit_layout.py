import itertools
cands = []
for s in range(16):
    for m in range(0, 14):
        bits = {s, m, m + 1, m + 2}
        if len(bits) < 4: continue
        a, b = 15 - s, 6 - m          # sign: left shift a >= 0; mag: shift b (b < 0 = right, |b| <= 7)
        if a < 0 or b < -7: continue
        cands.append((frozenset(bits), s, m, a, b))
best = []
for combo in itertools.combinations(cands, 4):
    used = set(); ok = True
    for c in combo:
        if used & c[0]: ok = False; break
        used |= c[0]
    if not ok or len(used) != 16: continue
    shifts = set()
    lop = 0
    for _, s, m, a, b in combo:
        if a: shifts.add(('L', a))
        if b > 0: shifts.add(('L', b))
        if b < 0: shifts.add(('R', -b))
        lop += 1 if a == b else 2
    cost = len(shifts) + lop
    best.append((cost, sorted((s, m, a, b) for _, s, m, a, b in combo), sorted(shifts)))
best.sort()
print("ops per 4 registers (shifts + LOP3), excluding HMUL2; current ~15")
for b in best[:6]: print(b)
