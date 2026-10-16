//! Bindings for `ed-emi`: a KNP-EMI axon bundle (Publication 2).
//!
//! [`EmiSim`] owns the finite element solver and turns its per-DOF state into what the
//! cell-scale view draws: membrane potential on the cell surfaces, one extracellular (or
//! intracellular) field on a plane cut, and φ_M traces at probes.

use ed_emi::mesh::{Aabb, Mesh, MeshSpec};

use crate::emi_slice::PlaneCut;
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

#[wasm_bindgen]
pub struct EmiSim {
    solver: EmiSolver,
    /// Axons per side of the bundle.
    per_side: usize,
    /// Membrane nodes of each probe: one per axon on the probe plane for the bundle.
    probe_nodes: Vec<Vec<usize>>,
    probe_x: f64,
    /// Interleaved `[t, φ_M of probe 1 … n]` per step since the last `take_traces`.
    traces: Vec<f32>,
    stimulated: u32,
    /// The last plane cut, by (axis, frac).
    cut: Option<(usize, f64, PlaneCut)>,
    /// Mesh bounds (lo, hi), µm.
    bounds: ([f64; 3], [f64; 3]),
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

    /// Moves the bundle's probes to the grid plane nearest `x` µm from the start of the
    /// axons (one per axon).
    pub fn set_probe(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let mesh = self.solver.mesh();
        let target = END_GAP + x.clamp(0.0, LENGTH);
        let xs = &mesh.grid.as_ref().expect("bundles have a grid").axes[0];
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

    /// Domain bounds `[x_lo, y_lo, z_lo, x_hi, y_hi, z_hi]`, µm.
    pub fn bounds(&self) -> Vec<f32> {
        let (lo, hi) = &self.bounds;
        lo.iter().chain(hi).map(|&v| v as f32).collect()
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

    /// Membrane triangles as membrane node indices, counter-clockwise from outside.
    pub fn membrane_triangles(&self) -> Vec<u32> {
        self.solver.mesh().membrane_facets.concat()
    }

    /// Membrane potential per membrane node, mV.
    pub fn phi_m(&self) -> Vec<f32> {
        self.solver.phi_m().iter().map(|&v| v as f32).collect()
    }

    /// Triangle corners (xyz each) of the plane cut normal to `axis` through fraction
    /// `frac` of the domain.
    pub fn slice_positions(&mut self, axis: usize, frac: f64) -> Vec<f32> {
        self.slice(axis, frac).positions.clone()
    }
    /// Subdomain tag of each slice triangle (0 = extracellular).
    pub fn slice_tags(&mut self, axis: usize, frac: f64) -> Vec<f32> {
        self.slice(axis, frac).tags.clone()
    }
    /// `field` ("phi" mV, or "na", "k", "cl" mM) at the slice triangle corners.
    pub fn slice_values(
        &mut self,
        axis: usize,
        frac: f64,
        field: &str,
    ) -> Result<Vec<f32>, JsError> {
        self.slice_field(axis, frac, field).map_err(js)
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
            bounds: mesh.bounds(),
            solver: EmiSolver::new(mesh, params),
            per_side: n,
            probe_nodes: Vec::new(),
            probe_x: 0.0,
            traces: Vec::new(),
            stimulated: 0,
            cut: None,
        };
        sim.set_probe(0.75 * LENGTH);
        Ok(sim)
    }

    fn slice_field(&mut self, axis: usize, frac: f64, field: &str) -> Result<Vec<f32>, String> {
        let k = match field {
            "phi" => None,
            "na" => Some(0),
            "k" => Some(1),
            "cl" => Some(2),
            _ => return Err(format!("unknown field {field:?}")),
        };
        self.slice(axis, frac);
        let data = k.map_or(self.solver.phi(), |k| self.solver.conc(k));
        Ok(self.cut.as_ref().expect("just cut").2.values(data))
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

    /// The plane cut normal to `axis` at fraction `frac` of the domain, cached.
    fn slice(&mut self, axis: usize, frac: f64) -> &PlaneCut {
        let axis = axis.min(2);
        let frac = if frac.is_finite() {
            frac.clamp(0.0, 1.0)
        } else {
            0.5
        };
        if !matches!(&self.cut, Some((a, f, _)) if *a == axis && *f == frac) {
            let (lo, hi) = self.bounds;
            // Keep the plane just inside the box so the end faces still cut tets.
            let mut at = lo[axis] + (hi[axis] - lo[axis]) * frac.clamp(1e-6, 1.0 - 1e-6);
            let mesh = self.solver.mesh();
            if let Some(grid) = &mesh.grid {
                // Midway through a grid layer, never on a membrane face.
                let xs = &grid.axes[axis];
                let i = xs.partition_point(|&p| p <= at).clamp(1, xs.len() - 1);
                at = 0.5 * (xs[i - 1] + xs[i]);
            }
            self.cut = Some((axis, frac, PlaneCut::new(mesh, axis, at)));
        }
        &self.cut.as_ref().expect("just cut").2
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
            self.traces.push((sum / nodes.len() as f64) as f32);
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
    fn membrane_triangles_cover_every_axon_surface() {
        let s = EmiSim::build(3, 0.1, 0.1).unwrap();
        let p = s.membrane_positions();
        let tris = s.membrane_triangles();
        let total: f64 = tris
            .chunks(3)
            .map(|t| {
                let q = |i: usize| [0, 1, 2].map(|k| f64::from(p[3 * t[i] as usize + k]));
                let (a, b, c) = (q(0), q(1), q(2));
                let u = [0, 1, 2].map(|k| b[k] - a[k]);
                let w = [0, 1, 2].map(|k| c[k] - a[k]);
                let x = [
                    u[1] * w[2] - u[2] * w[1],
                    u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0],
                ];
                0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
            })
            .sum();
        let one = Aabb::new([0.0; 3], [LENGTH, SIDE, SIDE]).surface_area();
        assert!((total / (9.0 * one) - 1.0).abs() < 1e-5, "{total} vs {one}");
    }

    #[test]
    fn slices_sample_the_right_side() {
        let mut s = EmiSim::build(3, 0.1, 0.1).unwrap();
        for (axis, frac) in [(0, 0.5), (2, 0.5), (1, 0.0), (2, 1.0)] {
            let tags = s.slice_tags(axis, frac);
            assert_eq!(s.slice_positions(axis, frac).len(), 9 * tags.len());
            let phi = s.slice_field(axis, frac, "phi").unwrap();
            assert_eq!(phi.len(), 3 * tags.len());
            // At rest φ_e = 0 outside and φ_i = rest potential inside.
            for (t, &tag) in tags.iter().enumerate() {
                let v = phi[3 * t];
                let ok = if tag == 0.0 {
                    v.abs() < 1e-6
                } else {
                    v < -50.0
                };
                assert!(ok, "tag {tag}: {v}");
            }
        }
        // A cross-section through the middle shows all nine axons.
        let tags = s.slice_tags(0, 0.5);
        assert!((1..=9).all(|c| tags.contains(&(c as f32))));
        assert!(s.slice_tags(2, 0.5).contains(&5.0));
        assert!(s.slice_field(2, 0.5, "q").is_err());
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
