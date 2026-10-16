"""Generates the realistic KNP-EMI cell geometries shipped with the app (issue #63).

Each geometry is a union of spheres, ellipsoids and (tapered) cylinders, meshed with gmsh into
tetrahedra that are fine at the membrane and coarsen with distance from it. Writes
`web/public/meshes/<name>.edmesh` (format in crates/ed-emi/src/meshfile.rs) and
`<name>.json` (stimulus sites and probes for the view).

    python3 -m pip install gmsh==4.15.2 numpy   # the Linux wheel needs system GL/X libraries
    python3 tools/meshes/generate.py            # all geometries
    python3 tools/meshes/generate.py neuron     # one: neuron, spiny-dendrite or hay-neuron (slow)

Units are µm. z is up.
"""

import json
import math
import sys
from pathlib import Path

import gmsh
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "web" / "public" / "meshes"


class Shape:
    """A cell as tubes (p0, p1, r0, r1) and balls (centre, r), with mesh size rules."""

    def __init__(self):
        self.tubes = []
        self.balls = []
        self.ellipsoids = []

    def tube(self, p0, p1, r0, r1=None):
        self.tubes.append((np.asarray(p0, float), np.asarray(p1, float), r0, r0 if r1 is None else r1))

    def ball(self, c, r):
        self.balls.append((np.asarray(c, float), r))

    def ellipsoid(self, c, axes):
        self.ellipsoids.append((np.asarray(c, float), np.asarray(axes, float)))

    def occ(self):
        occ = gmsh.model.occ
        vols = []
        for c, ax in self.ellipsoids:
            tag = occ.addSphere(*c, 1.0)
            # Turn the sphere's poles to ±y: a dendrite joining at a pole (often +z, the
            # apical trunk) gives flat tets there.
            occ.rotate([(3, tag)], *c, 1, 0, 0, math.pi / 2)
            occ.dilate([(3, tag)], *c, *ax)
            vols.append((3, tag))
        for p0, p1, r0, r1 in self.tubes:
            d = p1 - p0
            if abs(r0 - r1) < 1e-9:
                vols.append((3, occ.addCylinder(*p0, *d, r0)))
            else:
                vols.append((3, occ.addCone(*p0, *d, r0, r1)))
        for c, r in self.balls:
            vols.append((3, occ.addSphere(*c, r)))
        if len(vols) == 1:
            return vols
        fused, _ = occ.fuse(vols[:1], vols[1:])
        return fused

    def size(self, p, per_radius, ball_factor, growth, max_h=math.inf):
        """Target edge length at p: a fraction of the nearest part's radius (at most
        `max_h` at the membrane), growing linearly with distance from the membrane."""
        h = math.inf
        for p0, p1, r0, r1 in self.tubes:
            d = p1 - p0
            t = min(max(np.dot(p - p0, d) / np.dot(d, d), 0.0), 1.0)
            r = r0 + (r1 - r0) * t
            dist = max(np.linalg.norm(p - (p0 + t * d)) - r, 0.0)
            h = min(h, min(per_radius * r, max_h) + growth * dist)
        for c, r in self.balls:
            dist = max(np.linalg.norm(p - c) - r, 0.0)
            h = min(h, min(ball_factor * per_radius * r, max_h) + growth * dist)
        for c, ax in self.ellipsoids:
            # Distance to the surface, approximated through the scaled radius.
            q = np.linalg.norm((p - c) / ax)
            dist = max((q - 1.0) * ax.min(), 0.0)
            h = min(h, min(ball_factor * per_radius * ax.min(), max_h) + growth * dist)
        return h


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def neuron():
    """A stylised pyramidal neuron: soma, tapering apical dendrite, three basal dendrites
    and an axon. Diameters from typical cortical layer 5 cells, lengths shortened."""
    s = Shape()
    soma_r = 6.0
    s.ball([0, 0, 0], soma_r)
    parts = {
        "apical": ([0, 0, 1], 70.0, 1.3, 0.6),
        "basal1": ([0.8, 0.3, -0.5], 40.0, 0.7, 0.4),
        "basal2": ([-0.7, 0.5, -0.5], 40.0, 0.7, 0.4),
        "basal3": ([0.1, -0.9, -0.4], 40.0, 0.7, 0.4),
        "axon": ([0, 0, -1], 50.0, 0.45, 0.35),
    }
    tips = {}
    for name, (d, length, r0, r1) in parts.items():
        d = unit(d)
        p0 = d * (soma_r - 0.5)
        p1 = p0 + d * length
        s.tube(p0, p1, r0, r1)
        s.ball(p1, r1)  # rounded end
        tips[name] = p1 - d * 5.0
    meta = {
        "label": "Neuron",
        "description": "A stylised pyramidal neuron (soma 12 µm across, apical and basal dendrites, axon) "
        "in a box with 4 µm of extracellular space around it.",
        "stimuli": [
            {"key": "apical", "label": "Apical dendrite tip", "cell": 1, "centre": tips["apical"], "radius": 8.0, "g": 4.0},
            {"key": "basal", "label": "Basal dendrite tip", "cell": 1, "centre": tips["basal1"], "radius": 8.0, "g": 4.0},
            {"key": "soma", "label": "Soma", "cell": 1, "centre": [0, 0, 0], "radius": soma_r + 0.5, "g": 2.0},
        ],
        "probes": [
            {"label": "Soma", "centre": [0, 0, 0], "radius": soma_r + 0.5},
            {"label": "Apical tip", "centre": tips["apical"], "radius": 5.0},
            {"label": "Basal tip", "centre": tips["basal1"], "radius": 5.0},
            {"label": "Axon", "centre": tips["axon"], "radius": 5.0},
        ],
    }
    # Centred on the soma across (x, y), so the default lengthwise slice cuts through it.
    return s, meta, dict(margin=4.0, through=None, centred=(0, 1), per_radius=1.2, ball_factor=0.5, growth=1.0)


def spiny_dendrite():
    """A 20 µm dendrite with spines of the three classic shapes (thin, stubby,
    mushroom), dimensions from serial-section EM of hippocampal CA1 dendrites."""
    s = Shape()
    length, rd = 20.0, 0.5
    s.tube([0, 0, 0], [length, 0, 0], rd)
    # (x, angle around the shaft, neck radius, neck length, head radius)
    spines = [
        ("Thin spine", 3.0, 0.0, 0.05, 1.2, 0.2),
        ("Mushroom spine", 7.0, 2.4, 0.08, 0.8, 0.35),
        ("Stubby spine", 10.0, 4.4, 0.25, 0.3, 0.3),
        ("Mushroom spine 2", 13.0, 1.0, 0.08, 0.9, 0.3),
        ("Thin spine 2", 16.5, 3.4, 0.05, 1.0, 0.18),
    ]
    heads = []
    for label, x, ang, rn, ln, rh in spines:
        d = np.array([0.0, math.cos(ang), math.sin(ang)])
        p0 = np.array([x, 0, 0]) + d * (rd - 0.05)
        p1 = p0 + d * ln
        s.tube(p0, p1, rn)
        c = p1 + d * 0.8 * rh
        s.ball(c, rh)
        heads.append((label, c, rh, np.array([x, 0, 0]) + d * rd))
    meta = {
        "label": "Spiny dendrite",
        "description": "A 20 µm piece of dendrite (1 µm across) with thin, stubby and mushroom spines, "
        "in 1.5 µm of extracellular space. The shaft is sealed where it leaves the box.",
        "stimuli": [
            {"key": f"spine{i}", "label": f"{label} head", "cell": 1, "centre": c, "radius": rh + 0.05, "g": 100.0}
            for i, (label, c, rh, _) in enumerate(heads)
            if not label.endswith("2")
        ]
        + [{"key": "shaft", "label": "Dendrite shaft", "cell": 1, "centre": [length / 2, 0, 0], "radius": 1.0, "g": 20.0}],
        "probes": [{"label": f"{label} head", "centre": c, "radius": rh + 0.05} for label, c, rh, _ in heads[:2]]
        + [{"label": "Shaft under the mushroom spine", "centre": heads[1][3], "radius": 0.5}]
        + [{"label": "Shaft end", "centre": [length - 1, 0, 0], "radius": 0.7}],
    }
    return s, meta, dict(margin=1.5, through=(0, 0.0, length), per_radius=1.2, ball_factor=0.6, growth=0.5)


HAY = Path(__file__).resolve().parents[2] / "crates" / "ed-knp" / "data" / "hay_l5pc.txt"


def hay_segments():
    """(region, start, centre, end, diameter) per segment of the Hay et al. (2011) layer 5b
    pyramidal cell in crates/ed-knp/data (soma at the origin, apical axis +z). The
    diameter gives the segment's membrane area over its two chords."""
    segs = []
    for line in HAY.read_text().splitlines():
        f = line.split()
        if not f or f[0] not in ("soma", "basal", "apical", "axon"):
            continue
        vals = [float(x) for x in f[3:]]
        for k in range(int(f[2])):
            v = vals[12 * k : 12 * k + 12]
            a, c, b = np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9])
            length = np.linalg.norm(c - a) + np.linalg.norm(b - c)
            segs.append((f[0], a, c, b, v[9] / (math.pi * length)))
    return segs


MIN_DIAMETER = 0.8
# Smallest tet quality 6√2·V/ℓ_max³ allowed (1 for a regular tet).
MIN_QUALITY = 1e-3


def hay_neuron():
    """Soma and proximal dendrites of the reconstructed Hay et al. (2011) layer 5b pyramidal
    cell (ModelDB 139653, the morphology the tissue scale uses), cut by a box 60 µm
    across. Dendrites are sealed where they leave the box. Branches thinner than
    `MIN_DIAMETER` are left out: resolving them would need more elements than a browser
    can step in real time."""
    half = 30.0
    s = Shape()
    segs = hay_segments()
    soma = segs[0]
    # The soma: an ellipsoid with its length and mean width, aligned with x (the soma's axis
    # is within 3° of x).
    s.ellipsoid(soma[2], [np.linalg.norm(soma[3] - soma[1]) / 2, soma[4] / 2, soma[4] / 2])
    kept = [
        seg for seg in segs[1:]
        if min(np.abs(seg[1]).max(), np.abs(seg[3]).max()) <= half + 2 and seg[4] >= MIN_DIAMETER
    ]
    # NEURON joins each root section to the soma's midpoint, but its first point can lie
    # outside the ellipsoid. Start it inside instead, so the cell is one connected volume and
    # the tube crosses the soma surface at a steep angle (bridging tubes or balls that touch
    # the ellipsoid almost tangentially leave flat tets).
    centre, axes = soma[2], s.ellipsoids[0][1]
    ends = [seg[3] for seg in segs[1:]]
    for k, (region, a, c, b, d) in enumerate(kept):
        reach = np.linalg.norm((a - centre) / axes)  # 1 on the soma surface
        if reach > 0.7 and min(np.linalg.norm(a - e) for e in ends) > 1e-3:
            kept[k] = (region, centre + (a - centre) * 0.7 / reach, c, b, d)
    starts = [seg[1] for seg in kept]
    for _, a, c, b, d in kept:
        r = max(d / 2, 0.25)
        bent = min(np.linalg.norm(c - a), np.linalg.norm(b - c)) > 1e-3 and np.dot(
            unit(c - a), unit(b - c)
        ) < math.cos(math.radians(15))
        for p, q in ((a, c), (c, b)) if bent else ((a, b),):  # one tube if (nearly) straight
            s.tube(p, q, r)
        # A ball fills the gap where sections bend; at a straight joint it would only touch
        # the tubes tangentially (flat tets), so leave it out there.
        nxt = [k for k, p in enumerate(starts) if np.linalg.norm(p - b) < 1e-3]
        straight = len(nxt) == 1 and np.dot(unit(b - c), unit(kept[nxt[0]][2] - b)) > math.cos(math.radians(15))
        if not straight:
            s.ball(b, r)
    def on_cell(region, target):
        """The point of a meshed `region` segment nearest `target`, inside the box."""
        pts = [p for r, a, c, b, _ in kept if r == region for p in (a, c, b) if np.abs(p).max() < half - 2]
        return min(pts, key=lambda p: np.linalg.norm(p - np.asarray(target, float)))

    apical, axon = on_cell("apical", [0, 0, 22]), on_cell("axon", [0, 0, -22])
    basal = on_cell("basal", [20, 0, 0])
    meta = {
        "label": "Reconstructed neuron",
        "description": "The soma and proximal dendrites of a reconstructed layer 5b pyramidal cell "
        "(Hay et al. 2011, ModelDB 139653), the morphology used at the tissue scale. Dendrites "
        "are cut and sealed at the edges of the 60 µm box; branches thinner than 0.8 µm are left out.",
        "stimuli": [
            {"key": "apical", "label": "Apical trunk", "cell": 1, "centre": apical, "radius": 6.0, "g": 4.0},
            {"key": "basal", "label": "Basal dendrite", "cell": 1, "centre": basal, "radius": 8.0, "g": 10.0},
            {"key": "soma", "label": "Soma", "cell": 1, "centre": [0, 0, 0], "radius": 12.0, "g": 2.0},
        ],
        "probes": [
            {"label": "Soma", "centre": [0, 0, 0], "radius": 12.0},
            {"label": "Apical trunk", "centre": apical, "radius": 4.0},
            {"label": "Basal dendrite", "centre": basal, "radius": 4.0},
            {"label": "Axon", "centre": axon, "radius": 4.0},
        ],
    }
    return s, meta, dict(
        margin=0.0, box=(-half, half), per_radius=2.2, ball_factor=0.5, growth=2.0, max_h=3.0,
    )


GEOMETRIES = {"neuron": neuron, "spiny-dendrite": spiny_dendrite, "hay-neuron": hay_neuron}


def dump_json(meta):
    """JSON with one stimulus or probe per line, so the files stay short to diff."""
    one = lambda v: json.dumps(v, ensure_ascii=False)
    fields = []
    for k, v in meta.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            v = "[\n    " + ",\n    ".join(map(one, v)) + "\n  ]"
        else:
            v = one(v)
        fields.append(f"  {one(k)}: {v}")
    return "{\n" + ",\n".join(fields) + "\n}\n"


def build(name):
    shape, meta, opts = GEOMETRIES[name]()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    occ = gmsh.model.occ
    cell = shape.occ()
    occ.synchronize()
    if len(cell) != 1:
        sys.exit(f"{name}: the cell is {len(cell)} disconnected solids")
    box = np.array(gmsh.model.getBoundingBox(*cell[0]))
    lo, hi = box[:3] - opts["margin"], box[3:] + opts["margin"]
    if "box" in opts:  # a fixed cube that cuts the cell
        lo, hi = np.full(3, opts["box"][0]), np.full(3, opts["box"][1])
    for axis in opts.get("centred", ()):  # symmetric about the origin along these axes
        hi[axis] = max(-lo[axis], hi[axis])
        lo[axis] = -hi[axis]
    if opts.get("through") is not None:  # the cell leaves the box along this axis
        axis, a, b = opts["through"]
        lo[axis], hi[axis] = a, b
    ecs = occ.addBox(*lo, *(hi - lo))
    # Fragmenting makes the two volumes share the membrane surface (a conforming mesh);
    # parts of the cell outside the box are dropped.
    _, children = occ.fragment([(3, ecs)], cell)
    occ.synchronize()
    cell_volumes = {t for part in children[1:] for _, t in part}
    inside_box = lambda t: all(
        lo[k] - 1e-6 <= v <= hi[k] + 1e-6
        for k, v in enumerate(gmsh.model.occ.getCenterOfMass(3, t))
    )
    for _, t in gmsh.model.getEntities(3):
        if not inside_box(t):
            gmsh.model.removeEntities([(3, t)])  # not recursive: the cut faces are shared
    n_cell_volumes = sum(t in cell_volumes for _, t in gmsh.model.getEntities(3))
    for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{opt}", 0)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    gmsh.model.mesh.setSizeCallback(
        lambda dim, tag, x, y, z, lc: shape.size(
            np.array([x, y, z]), opts["per_radius"], opts["ball_factor"], opts["growth"], opts.get("max_h", math.inf)
        )
    )
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(node_tags)}
    tets, tags = [], []
    for _, vol in gmsh.model.getEntities(3):
        _, _, conn = gmsh.model.mesh.getElements(3, vol)
        t = np.array([index[int(n)] for n in conn[0]], dtype=np.uint32).reshape(-1, 4)
        tets.append(t)
        tags.append(np.full(len(t), 1 if vol in cell_volumes else 0, dtype=np.uint16))
    gmsh.finalize()
    points = coords.reshape(-1, 3)
    tets, tags = np.concatenate(tets), np.concatenate(tags)
    if len(set(int(t) for t in np.unique(tags[tags > 0]))) != 1 or n_cell_volumes != 1:
        sys.exit(f"{name}: the box cuts the cell into {n_cell_volumes} pieces")
    p = points[tets]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6
    edges = [np.linalg.norm(p[:, i] - p[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)]
    quality = 6 * math.sqrt(2) * vol / np.max(edges, axis=0) ** 3  # 1 for a regular tet
    if quality.min() < MIN_QUALITY:
        worst = p[np.argmin(quality)].mean(axis=0).round(2).tolist()
        sys.exit(f"{name}: {np.sum(quality < MIN_QUALITY)} near-flat tets (worst {quality.min():.1e} at {worst})")
    # Drop points no tet uses and renumber.
    used = np.unique(tets)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points, tets = points[used], remap[tets]

    short = len(points) <= 65536
    path = OUT / f"{name}.edmesh"
    with open(path, "wb") as f:
        f.write(b"EDMESH01")
        np.array([len(points), len(tets), 1, 1 if short else 0], dtype="<u4").tofile(f)
        points.astype("<f4").tofile(f)
        tets.astype("<u2" if short else "<u4").tofile(f)
        tags.astype("<u2").tofile(f)
    meta.update(
        file=path.name,
        extent={"lo": [round(float(v), 4) for v in lo], "hi": [round(float(v), 4) for v in hi]},
        points=len(points),
        tets=len(tets),
        gmsh=gmsh.__version__,
    )
    to_list = lambda v: [round(float(x), 4) for x in v]
    for item in meta["stimuli"] + meta["probes"]:
        item["centre"] = to_list(item["centre"])
        item["radius"] = round(float(item["radius"]), 4)
    (OUT / f"{name}.json").write_text(dump_json(meta), encoding="utf-8")
    print(f"{name}: {len(points)} points, {len(tets)} tets ({int(tags.sum())} in the cell), {path.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or GEOMETRIES:
        build(name)
