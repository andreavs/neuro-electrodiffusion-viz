#!/usr/bin/env python3
"""Generate hay_l5pc.txt from the Hay et al. (2011) morphology `cell1.asc`.

Usage: python3 gen_hay_l5pc.py cell1.asc > hay_l5pc.txt

`cell1.asc` is in ModelDB entry 139653 (morphologies/cell1.asc, also at
https://github.com/ModelDBRepository/139653). The output follows NEURON's
Import3d and the L5PCtemplate.hoc rules: one section per unbranched piece,
a child section starts at its parent's last 3-D point, roots attach to
soma(0.5), nseg = 1 + 2 int(L/40), and the axon is replaced by two 30 µm,
1 µm sections. Simplification: the soma contour becomes one cylinder with
length = diameter = twice the mean contour radius (the area of a sphere).

Coordinates are µm with the soma centroid at the origin and the apical axis
along +z (file (x, y, z) -> (x, -z, y)).
"""
import math
import re
import sys

SEG_LEN = 40.0  # nseg = 1 + 2 int(L / SEG_LEN)


def tokens(text):
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r'"[^"]*"|\(|\)|\||[^\s()|]+', text)


def parse(toks):
    """S-expressions; each list is split into alternatives at '|'."""
    pos = 0

    def node():
        nonlocal pos
        pos += 1  # '('
        alts, cur = [], []
        while toks[pos] != ")":
            t = toks[pos]
            if t == "(":
                cur.append(node())
                continue
            if t == "|":
                alts.append(cur)
                cur = []
            else:
                cur.append(t)
            pos += 1
        pos += 1
        alts.append(cur)
        return alts

    out = []
    while pos < len(toks):
        if toks[pos] == "(":
            out.append(node())
        else:
            pos += 1
    return out


def number(s):
    try:
        return float(s)
    except ValueError:
        return None


def point(n):
    """(x y z d [S1]) -> [x, y, z, d], else None."""
    if len(n) != 1 or len(n[0]) < 4 or not all(isinstance(x, str) for x in n[0][:4]):
        return None
    v = [number(x) for x in n[0][:4]]
    return v if None not in v else None


def word(n):
    if len(n) == 1 and n[0] and isinstance(n[0][0], str) and number(n[0][0]) is None:
        return n[0][0].strip('"')
    return None


def branch(alt):
    """A list of points then optionally a split into child branches."""
    pts, children = [], []
    for it in alt:
        if isinstance(it, str) or word(it) is not None:
            continue
        p = point(it)
        if p:
            pts.append(p)
        elif len(it) >= 2:
            children += [branch(a) for a in it]
    return pts, children


def transform(p, c):
    x, y, z = (p[k] - c[k] for k in range(3))
    return [x, -z, y, p[3]]


def segments(pts, nseg):
    """Per segment: (area µm², ∫ds/d² 1/µm, centre arc fraction) and the
    3-D points at arc positions, integrating the piecewise-linear profile."""
    arc = [0.0]
    for a, b in zip(pts, pts[1:]):
        arc.append(arc[-1] + math.dist(a[:3], b[:3]))
    total = arc[-1]

    def at(s):
        for i in range(1, len(pts)):
            if s <= arc[i] or i == len(pts) - 1:
                l = arc[i] - arc[i - 1]
                f = 0.0 if l == 0 else min(max((s - arc[i - 1]) / l, 0.0), 1.0)
                return [pts[i - 1][k] + f * (pts[i][k] - pts[i - 1][k]) for k in range(4)]

    out = []
    for j in range(nseg):
        s0, s1 = total * j / nseg, total * (j + 1) / nseg
        cuts = [s0] + [s for s in arc if s0 < s < s1] + [s1]
        area = k = 0.0
        for u, v in zip(cuts, cuts[1:]):
            d0, d1 = at(u)[3], at(v)[3]
            area += math.pi * 0.5 * (d0 + d1) * (v - u)
            k += (v - u) / (d0 * d1)
        out.append((area, k, at(s0)[:3], at(0.5 * (s0 + s1))[:3], at(s1)[:3]))
    return total, out


def main(path):
    soma, roots = None, []
    for t in parse(tokens(open(path).read())):
        alt = t[0]
        words = [word(x) for x in alt if not isinstance(x, str)]
        if "CellBody" in words:
            soma = [p for p in (point(x) for x in alt if not isinstance(x, str)) if p]
        for kind, region in (("Dendrite", "basal"), ("Apical", "apical")):
            if kind in words:
                roots.append((region, branch(alt)))
    c = [sum(p[k] for p in soma) / len(soma) for k in range(3)]
    r = sum(math.dist(p[:3], c) for p in soma) / len(soma)

    # region parent nseg, then per segment: start xyz, centre xyz, end xyz,
    # area, ∫ds/d², path distance from soma(0.5) to the centre.
    lines = []

    def emit(region, parent, rows):
        flat = [x for row in rows for x in row]
        body = " ".join(f"{x:.5g}" for x in flat)
        lines.append(f"{region} {parent} {len(rows)} {body}")
        return len(lines) - 1

    emit("soma", -1, [[0, 0, -r, 0, 0, 0, 0, 0, r, 4 * math.pi * r * r, 1 / (2 * r), 0]])
    max_apical = 0.0

    def walk(region, pts, children, parent, parent_end, d0):
        nonlocal max_apical
        pts = [transform(p, c) for p in pts]
        if parent_end is not None:
            pts = [parent_end] + pts
        total, segs = segments(pts, 0)  # length only
        nseg = 1 + 2 * int(total / SEG_LEN)
        total, segs = segments(pts, nseg)
        rows = [
            [*s0, *m, *s1, a, k, d0 + total * (j + 0.5) / nseg]
            for j, (a, k, s0, m, s1) in enumerate(segs)
        ]
        me = emit(region, parent, rows)
        if region == "apical" and not children:
            max_apical = max(max_apical, d0 + total)
        for cp, cc in children:
            walk(region, cp, cc, me, pts[-1], d0 + total)

    for region, (pts, children) in roots:
        walk(region, pts, children, 0, None, 0.0)
    for i in range(2):
        z0, z1 = -r - 30 * i, -r - 30 * (i + 1)
        emit("axon", 0 if i == 0 else len(lines) - 1,
             [[0, 0, z0, 0, 0, 0.5 * (z0 + z1), 0, 0, z1, 30 * math.pi, 30.0, 30 * i + 15]])

    print(__doc__.strip().replace("\n", "\n# ").replace('"""', "").join(["# ", ""]))
    print("# Columns: region, parent section (-1 for the soma), nseg, then per segment:")
    print("# start xyz, centre xyz, end xyz, area µm², ∫ds/d² 1/µm, path distance µm.")
    print(f"max_apical {max_apical:.2f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
