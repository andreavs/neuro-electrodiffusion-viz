//! Bindings for `ed-emi`: a KNP-EMI axon bundle (Publication 2).
//!
//! [`EmiSim`] owns the finite element solver and turns its per-DOF state into what the
//! cell-scale view draws: membrane potential on the axon surfaces, one extracellular (or
//! intracellular) field on a slice through the bundle, and φ_M traces at a probe position.

use ed_emi::mesh::{Aabb, Mesh, MeshSpec, NONE};
use ed_emi::solver::{EmiParams, EmiSolver};
use wasm_bindgen::prelude::*;

/// Axon length, side and the extracellular space around the bundle and past its ends, µm.
const LENGTH: f64 = 200.0;
const SIDE: f64 = 0.2;
const MARGIN: f64 = 0.3;
const END_GAP: f64 = 5.0;
/// Synapse used as the stimulus: the first `STIM_LENGTH` µm of an axon, mS/cm².
const STIM_LENGTH: f64 = 25.0;
const STIM_G: f64 = 4.0;
/// Trace samples kept between two frames (oldest dropped first).
const MAX_TRACE_SAMPLES: usize = 4096;

const MAX_PER_SIDE: u32 = 4;
const GAP_RANGE: (f64, f64) = (0.05, 1.0);
const MAX_DT: f64 = 0.5;

/// An `n × n` bundle of square axons along x with `gap` µm between neighbours. Cell tag
/// `1 + iy + n iz` sits in column `iy`, row `iz`.
pub fn bundle_spec(n: usize, gap: f64) -> MeshSpec {
    let mut cells = Vec::with_capacity(n * n);
    for iz in 0..n {
        for iy in 0..n {
            let y0 = MARGIN + iy as f64 * (SIDE + gap);
            let z0 = MARGIN + iz as f64 * (SIDE + gap);
            cells.push(Aabb::new(
                [END_GAP, y0, z0],
                [END_GAP + LENGTH, y0 + SIDE, z0 + SIDE],
            ));
        }
    }
    let width = 2.0 * MARGIN + n as f64 * SIDE + (n - 1) as f64 * gap;
    MeshSpec {
        extent: [LENGTH + 2.0 * END_GAP, width, width],
        cells,
        // As `MeshSpec::default_bundle`: coarse along the axons, cell faces across.
        max_h: [5.0, 0.3, 0.3],
    }
}

/// Cell tags picked by a stimulus selection ("centre", "corner" or "all").
pub fn selection(n: usize, which: &str) -> Result<Vec<u16>, String> {
    let tag = |iy: usize, iz: usize| (1 + iy + n * iz) as u16;
    let mid = (n - 1) / 2;
    match which {
        "centre" => Ok(vec![tag(mid, mid)]),
        "corner" => Ok(vec![tag(0, 0)]),
        "all" => Ok((1..=(n * n) as u16).collect()),
        _ => Err(format!("unknown stimulus selection {which:?}")),
    }
}

/// A plane through the middle of one layer of hexes: one quad per hex of the layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slice {
    /// Normal axis (0 = x, 1 = y, 2 = z).
    pub axis: usize,
    /// Hex layer along `axis`.
    pub layer: usize,
}

impl Slice {
    /// The layer containing the fraction `frac` ∈ [0, 1] of the domain along `axis`.
    pub fn at(mesh: &Mesh, axis: usize, frac: f64) -> Self {
        let n = mesh.axes[axis].len() - 1;
        let x =
            mesh.axes[axis][0] + frac.clamp(0.0, 1.0) * (mesh.axes[axis][n] - mesh.axes[axis][0]);
        let layer = mesh.axes[axis].partition_point(|&p| p <= x).clamp(1, n) - 1;
        Self { axis, layer }
    }

    /// The two in-plane axes, in increasing order.
    fn plane_axes(self) -> (usize, usize) {
        match self.axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        }
    }

    /// Calls `f(hex, corners)` for each quad, `u` fastest, with each corner as the grid
    /// vertices on the two faces of the layer; corners go (u,v), (u+1,v), (u+1,v+1), (u,v+1).
    fn for_each_quad(self, mesh: &Mesh, mut f: impl FnMut(usize, [[usize; 2]; 4])) {
        let dims = mesh.vertex_dims();
        let (ua, va) = self.plane_axes();
        let vert = |idx: [usize; 3]| idx[0] + dims[0] * (idx[1] + dims[1] * idx[2]);
        let hex = |idx: [usize; 3]| idx[0] + (dims[0] - 1) * (idx[1] + (dims[1] - 1) * idx[2]);
        for iv in 0..dims[va] - 1 {
            for iu in 0..dims[ua] - 1 {
                let mut base = [0; 3];
                base[ua] = iu;
                base[va] = iv;
                base[self.axis] = self.layer;
                let corner = |du: usize, dv: usize| {
                    let mut p = base;
                    p[ua] += du;
                    p[va] += dv;
                    let mut q = p;
                    q[self.axis] += 1;
                    [vert(p), vert(q)]
                };
                f(
                    hex(base),
                    [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)],
                );
            }
        }
    }

    /// Quad corner positions (xyz per corner, 4 corners per quad), µm.
    pub fn positions(self, mesh: &Mesh) -> Vec<f32> {
        let mut out = Vec::new();
        self.for_each_quad(mesh, |_, corners| {
            for [a, b] in corners {
                let (pa, pb) = (mesh.vertex_position(a), mesh.vertex_position(b));
                out.extend((0..3).map(|k| (0.5 * (pa[k] + pb[k])) as f32));
            }
        });
        out
    }

    /// Subdomain tag of each quad (0 = extracellular).
    pub fn tags(self, mesh: &Mesh) -> Vec<f32> {
        let mut out = Vec::new();
        self.for_each_quad(mesh, |h, _| out.push(f32::from(mesh.hex_tag[h])));
        out
    }

    /// `field` per DOF, sampled at the quad corners from the quad's own subdomain.
    pub fn values(self, mesh: &Mesh, field: &[f64]) -> Vec<f32> {
        let mut out = Vec::new();
        self.for_each_quad(mesh, |h, corners| {
            let map = if mesh.hex_tag[h] == 0 {
                &mesh.vertex_outer
            } else {
                &mesh.vertex_inner
            };
            for [a, b] in corners {
                let (da, db) = (map[a], map[b]);
                debug_assert!(da != NONE && db != NONE);
                out.push((0.5 * (field[da as usize] + field[db as usize])) as f32);
            }
        });
        out
    }
}

/// Membrane quads (4 membrane node indices each): every hex face between a cell and the
/// extracellular space.
pub fn membrane_quads(mesh: &Mesh) -> Vec<u32> {
    let dims = mesh.vertex_dims();
    let mut node_of_vertex = vec![NONE; mesh.vertex_outer.len()];
    for (p, m) in mesh.membrane.iter().enumerate() {
        node_of_vertex[mesh.dof_vertex[m.inner as usize] as usize] = p as u32;
    }
    let h = [dims[0] - 1, dims[1] - 1, dims[2] - 1];
    let hex = |i: [usize; 3]| i[0] + h[0] * (i[1] + h[1] * i[2]);
    let vert = |i: [usize; 3]| i[0] + dims[0] * (i[1] + dims[1] * i[2]);
    let mut quads = Vec::new();
    for (hx, &tag) in mesh.hex_tag.iter().enumerate() {
        if tag == 0 {
            continue;
        }
        let idx = [hx % h[0], hx / h[0] % h[1], hx / (h[0] * h[1])];
        for (a, side) in (0..3).flat_map(|a| [(a, 0), (a, 1)]) {
            // Neighbour across the face; a cell face on the box boundary has no
            // extracellular side and so no membrane.
            let mut nb = idx;
            match (idx[a] + 2 * side).checked_sub(1) {
                Some(i) if i < h[a] => nb[a] = i,
                _ => continue,
            }
            if mesh.hex_tag[hex(nb)] != 0 {
                continue;
            }
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            // Counter-clockwise seen from outside the cell.
            let ccw = [(0, 0), (1, 0), (1, 1), (0, 1)];
            let order = if side == 1 {
                ccw
            } else {
                ccw.map(|(b, c)| (c, b))
            };
            let face = order.map(|(db, dc)| {
                let mut v = idx;
                v[a] += side;
                v[b] += db;
                v[c] += dc;
                node_of_vertex[vert(v)]
            });
            if face.iter().all(|&n| n != NONE) {
                quads.extend(face);
            }
        }
    }
    quads
}

#[wasm_bindgen]
pub struct EmiSim {
    solver: EmiSolver,
    per_side: usize,
    /// Membrane nodes of each cell on the probe plane.
    probe_nodes: Vec<Vec<usize>>,
    probe_x: f64,
    /// Interleaved `[t, φ_M of cell 1 … n]` per step since the last `take_traces`.
    traces: Vec<f32>,
    stimulated: u32,
}

#[wasm_bindgen]
impl EmiSim {
    /// `per_side × per_side` axons `gap` µm apart, time step `dt` ms.
    #[wasm_bindgen(constructor)]
    pub fn new(per_side: u32, gap: f64, dt: f64) -> Result<EmiSim, JsError> {
        Self::build(per_side, gap, dt).map_err(js)
    }

    /// Opens the stimulus synapse now on the axons picked by `which` ("centre", "corner",
    /// "all").
    pub fn stimulate(&mut self, which: &str) -> Result<(), JsError> {
        self.stimulate_cells(which).map_err(js)
    }

    pub fn set_dt(&mut self, dt: f64) -> Result<(), JsError> {
        check_dt(dt).map_err(js)?;
        self.solver.set_dt(dt);
        Ok(())
    }

    /// Moves the probe to the grid plane nearest `x` µm from the start of the axons.
    pub fn set_probe(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let mesh = self.solver.mesh();
        let target = END_GAP + x.clamp(0.0, LENGTH);
        let xs = &mesh.axes[0];
        let x = xs
            .iter()
            .copied()
            .min_by(|a, b| (a - target).abs().total_cmp(&(b - target).abs()))
            .unwrap_or(target);
        let mut nodes = vec![Vec::new(); mesh.n_cells];
        for (p, m) in mesh.membrane.iter().enumerate() {
            let px = mesh.vertex_position(mesh.dof_vertex[m.inner as usize] as usize)[0];
            if px == x {
                nodes[m.cell as usize - 1].push(p);
            }
        }
        self.probe_nodes = nodes;
        self.probe_x = x;
        self.traces.clear();
        self.record();
    }

    /// Takes up to `n` time steps. On failure the state is unusable and a new `EmiSim` is
    /// needed.
    pub fn step(&mut self, n: usize) -> Result<usize, JsError> {
        for _ in 0..n {
            if let Err(e) = self.solver.step() {
                return Err(js(format!(
                    "the cell-scale solver failed at t = {:.2} ms ({e}); try a smaller time step and press Reset",
                    self.solver.t()
                )));
            }
            self.record();
        }
        Ok(n)
    }

    pub fn time_ms(&self) -> f64 {
        self.solver.t()
    }
    pub fn n_cells(&self) -> usize {
        self.solver.mesh().n_cells
    }
    /// Probe position, µm from the start of the axons.
    pub fn probe_x(&self) -> f64 {
        self.probe_x - END_GAP
    }
    /// Bit `c − 1` is set for every cell `c` stimulated so far.
    pub fn stimulated(&self) -> u32 {
        self.stimulated
    }

    /// Domain size `[x, y, z]`, µm.
    pub fn extent(&self) -> Vec<f32> {
        let a = &self.solver.mesh().axes;
        a.iter()
            .map(|ax| *ax.last().unwrap_or(&0.0) as f32)
            .collect()
    }

    /// Membrane node positions, interleaved xyz in µm.
    pub fn membrane_positions(&self) -> Vec<f32> {
        let mesh = self.solver.mesh();
        mesh.membrane
            .iter()
            .flat_map(|m| mesh.vertex_position(mesh.dof_vertex[m.inner as usize] as usize))
            .map(|v| v as f32)
            .collect()
    }

    /// See [`membrane_quads`].
    pub fn membrane_quads(&self) -> Vec<u32> {
        membrane_quads(self.solver.mesh())
    }

    /// Membrane potential per membrane node, mV.
    pub fn phi_m(&self) -> Vec<f32> {
        self.solver.phi_m().iter().map(|&v| v as f32).collect()
    }

    /// Quad corners of the slice normal to `axis` through fraction `frac` of the domain.
    pub fn slice_positions(&self, axis: usize, frac: f64) -> Vec<f32> {
        self.slice(axis, frac).positions(self.solver.mesh())
    }
    pub fn slice_tags(&self, axis: usize, frac: f64) -> Vec<f32> {
        self.slice(axis, frac).tags(self.solver.mesh())
    }
    /// `field` ("phi" mV, or "na", "k", "cl" mM) at the slice quad corners.
    pub fn slice_values(&self, axis: usize, frac: f64, field: &str) -> Result<Vec<f32>, JsError> {
        let data = match field {
            "phi" => self.solver.phi(),
            "na" => self.solver.conc(0),
            "k" => self.solver.conc(1),
            "cl" => self.solver.conc(2),
            _ => return Err(js(format!("unknown field {field:?}"))),
        };
        Ok(self.slice(axis, frac).values(self.solver.mesh(), data))
    }

    /// Drains the trace samples recorded since the last call: `[t, φ_M(1) … φ_M(n)]` each.
    pub fn take_traces(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.traces)
    }
}

impl EmiSim {
    fn build(per_side: u32, gap: f64, dt: f64) -> Result<Self, String> {
        if !(1..=MAX_PER_SIDE).contains(&per_side) {
            return Err(format!("axons per side must be 1 to {MAX_PER_SIDE}"));
        }
        if !(GAP_RANGE.0..=GAP_RANGE.1).contains(&gap) {
            return Err(format!("gap must be {} to {} µm", GAP_RANGE.0, GAP_RANGE.1));
        }
        check_dt(dt)?;
        let n = per_side as usize;
        let mesh = Mesh::build(&bundle_spec(n, gap)).map_err(|e| e.to_string())?;
        let params = EmiParams {
            dt,
            ..EmiParams::default()
        };
        let mut sim = Self {
            solver: EmiSolver::new(mesh, params),
            per_side: n,
            probe_nodes: Vec::new(),
            probe_x: 0.0,
            traces: Vec::new(),
            stimulated: 0,
        };
        sim.set_probe(0.75 * LENGTH);
        Ok(sim)
    }

    fn stimulate_cells(&mut self, which: &str) -> Result<(), String> {
        let t = self.solver.t();
        for c in selection(self.per_side, which)? {
            self.solver
                .stimulate(c, 0.0, END_GAP + STIM_LENGTH, STIM_G, t);
            self.stimulated |= 1 << (c - 1);
        }
        Ok(())
    }

    fn slice(&self, axis: usize, frac: f64) -> Slice {
        Slice::at(self.solver.mesh(), axis.min(2), frac)
    }

    /// Appends one trace sample for the current time.
    fn record(&mut self) {
        let width = 1 + self.probe_nodes.len();
        if self.traces.len() >= MAX_TRACE_SAMPLES * width {
            self.traces.drain(..width);
        }
        self.traces.push(self.solver.t() as f32);
        let phi_m = self.solver.phi_m();
        for nodes in &self.probe_nodes {
            let sum: f64 = nodes.iter().map(|&p| phi_m[p]).sum();
            self.traces.push((sum / nodes.len().max(1) as f64) as f32);
        }
    }
}

fn check_dt(dt: f64) -> Result<(), String> {
    if dt.is_finite() && dt > 0.0 && dt <= MAX_DT {
        Ok(())
    } else {
        Err(format!("time step must be in (0, {MAX_DT}] ms"))
    }
}

fn js(msg: String) -> JsError {
    JsError::new(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membrane_quads_cover_every_axon_surface() {
        let mesh = Mesh::build(&bundle_spec(3, 0.1)).unwrap();
        let quads = membrane_quads(&mesh);
        let area = |q: [u32; 4]| {
            let p = q.map(|n| {
                mesh.vertex_position(
                    mesh.dof_vertex[mesh.membrane[n as usize].inner as usize] as usize,
                )
            });
            let d = |a: [f64; 3], b: [f64; 3]| {
                ((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>()).sqrt()
            };
            d(p[0], p[1]) * d(p[1], p[2])
        };
        let total: f64 = quads.chunks(4).map(|q| area(q.try_into().unwrap())).sum();
        let one = Aabb::new([0.0; 3], [LENGTH, SIDE, SIDE]).surface_area();
        assert!((total / (9.0 * one) - 1.0).abs() < 1e-6, "{total} vs {one}");
    }

    #[test]
    fn slices_have_matching_sizes_and_sample_the_right_side() {
        let s = EmiSim::build(3, 0.1, 0.1).unwrap();
        let mesh = s.solver.mesh();
        for (axis, frac) in [(0, 0.5), (2, 0.5), (1, 0.0), (2, 1.0)] {
            let sl = Slice::at(mesh, axis, frac);
            let tags = sl.tags(mesh);
            assert_eq!(sl.positions(mesh).len(), 12 * tags.len());
            let phi = sl.values(mesh, s.solver.phi());
            assert_eq!(phi.len(), 4 * tags.len());
            // At rest φ_e = 0 outside and φ_i = rest potential inside.
            for (q, &t) in tags.iter().enumerate() {
                let v = phi[4 * q];
                assert!(
                    if t == 0.0 { v.abs() < 1e-6 } else { v < -50.0 },
                    "tag {t}: {v}"
                );
            }
        }
        // A cross-section through the middle shows all nine axons.
        let tags = Slice::at(mesh, 0, 0.5).tags(mesh);
        assert!((1..=9).all(|c| tags.contains(&(c as f32))));
        // The middle of z cuts through the middle row of axons.
        assert!(Slice::at(mesh, 2, 0.5).tags(mesh).contains(&5.0));
    }

    #[test]
    fn centre_stimulus_fires_an_action_potential_past_the_probe() {
        let mut s = EmiSim::build(3, 0.1, 0.1).unwrap();
        s.stimulate_cells("centre").unwrap();
        assert_eq!(s.stimulated, 1 << 4);
        assert_eq!(s.take_traces().len(), 10); // the initial sample
        let mut peak = [f64::NEG_INFINITY; 9];
        for _ in 0..40 {
            s.step(1).unwrap();
            let tr = s.take_traces();
            assert_eq!(tr.len(), 10);
            for c in 0..9 {
                peak[c] = peak[c].max(f64::from(tr[1 + c]));
            }
        }
        assert!(peak[4] > 0.0, "centre axon spiked: {peak:?}");
        assert!(peak[0] < -40.0, "a neighbour did not: {peak:?}");
    }

    #[test]
    fn bad_input_is_rejected() {
        for (n, g, dt) in [
            (0, 0.1, 0.1),
            (5, 0.1, 0.1),
            (2, 0.0, 0.1),
            (2, 0.1, f64::NAN),
        ] {
            assert!(EmiSim::build(n, g, dt).is_err());
        }
        assert!(selection(2, "row").is_err());
        assert_eq!(selection(2, "centre").unwrap(), vec![1]);
        assert_eq!(selection(4, "all").unwrap().len(), 16);
    }
}
