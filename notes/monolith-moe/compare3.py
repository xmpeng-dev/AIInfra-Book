import re

def parse(p):
    out = {}
    pat = re.compile(r'iteration\s+(\d+)/\s*20.*?lm loss:\s+([0-9.E+-]+).*?grad norm:\s+([0-9.]+)')
    tpat = re.compile(r'TFLOP/s/GPU\):\s+([0-9.]+)/')
    for ln in open(p):
        m = pat.search(ln)
        if m:
            t = tpat.search(ln)
            out[int(m.group(1))] = (
                float(m.group(2)),
                float(m.group(3)),
                float(t.group(1)) if t else 0,
            )
    return out


b = parse('raw/baseline_rank7.log')
e = parse('raw/mmoe_rank7.log')
d = parse('raw/mmoe_dec_rank7.log')

print('iter | baseline   | eager      | decomposed | dec-base  | dec-eager | b gn  | d gn  | b TFLOP | e TFLOP | d TFLOP')
print('-----+------------+------------+------------+-----------+-----------+-------+-------+---------+---------+--------')
for it in sorted(b.keys() & e.keys() & d.keys()):
    bl, bg, bt = b[it]
    el, eg, et = e[it]
    dl, dg, dt = d[it]
    print(
        f'{it:4d} | {bl:10.6f} | {el:10.6f} | {dl:10.6f} | '
        f'{dl - bl:+9.2e} | {dl - el:+9.2e} | '
        f'{bg:5.2f} | {dg:5.2f} | {bt:7.1f} | {et:7.1f} | {dt:6.1f}'
    )
