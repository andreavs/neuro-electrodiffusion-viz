//! The KNP-EMI time step on a [`VoxelGeometry`].

use ed_core::constants::psi_mv;
use ed_core::linalg::{
    BoundaryConditions, CgOptions, DiffusionOperator, Face, Jacobi, LinearOperator, NullSpace, Pcg,
};
use ed_core::membrane::MembranePatches;
use ed_core::Simulation;

use crate::geometry::{MembraneFace, VoxelGeometry, ECS};
use crate::params::{EmiParams, StepStats, SPECIES, UA_PER_CM2_TO_FLUX};
use crate::precond::CablePreconditioner;

/// Steps between refreshes of the potential preconditioner.
const PRECONDITIONER_REFRESH: u64 = 10;

/// A scheduled synaptic input.
#[derive(Clone, Debug, PartialEq)]
struct SynapseEvent {
    t0: f64,
    g: f64,
    patches: Vec<usize>,
}

/// KNP-EMI solver state. See the crate documentation for the formulation.
pub struct EmiSolver {
    geo: VoxelGeometry,
    params: EmiParams,
    faces: Vec<MembraneFace>,
    /// Membrane state, one patch per entry of `faces`.
    pub patches: MembranePatches,
    /// Concentrations per species (see [`SPECIES`]) and voxel, mM.
    conc: [Vec<f64>; 3],
    /// Electric potential per voxel, mV; the extracellular mean is kept at 0.
    phi: Vec<f64>,
    /// Charge density Σ z c per voxel, mM (zero up to the splitting error).
    rho: Vec<f64>,
    /// Total membrane current I_M of the last step per face, µA/cm².
    i_m: Vec<f64>,
    time: f64,
    steps: u64,
    /// Ion amount per species held as membrane charge, mM·µm³ (see [`total_ions`](Self::total_ions)).
    membrane_store: [f64; 3],
    /// Pending synaptic events, sorted by onset.
    events: Vec<SynapseEvent>,
    psi: f64,

    pot_op: DiffusionOperator,
    pot_mg: CablePreconditioner,
    conc_ops: [DiffusionOperator; 3],
    conc_pre: [Jacobi; 3],
    pcg: Pcg,
    sigma: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
    delta: Vec<f64>,
    /// Faces between voxels of the same region: (lower, upper, 1/h²).
    bulk_faces: Vec<(u32, u32, f64)>,
    face_work: Vec<[f64; 2]>,
    /// Potential-operator face overrides: membrane faces first, then cell contacts.
    pot_faces: Vec<(Face, f64)>,
    /// Membrane coupling `I_M / (φ_M − φ_M*)` per face, µA/(cm²·mV).
    coupling: Vec<f64>,
    stats: StepStats,
}

impl EmiSolver {
    /// Initialises the geometry at rest: uniform intra/extracellular concentrations,
    /// patches at the membrane resting state, φ = φ_M inside cells and 0 outside.
    pub fn new(geo: VoxelGeometry, params: EmiParams) -> Self {
        assert!(params.dt > 0.0 && params.substeps > 0);
        let n = geo.grid.len();
        let faces = geo.membrane_faces();
        let mut cells_touching_ecs = vec![false; geo.voxel_counts().len()];
        faces
            .iter()
            .for_each(|f| cells_touching_ecs[f.cell as usize] = true);
        for (tag, &count) in geo.voxel_counts().iter().enumerate().skip(1) {
            assert!(
                count == 0 || cells_touching_ecs[tag],
                "cell {tag} has no membrane facing the extracellular space"
            );
        }
        let mut patches = MembranePatches::new(faces.len(), params.membrane, params.initial);
        patches.use_gate_table(true);
        let phi_rest = patches.phi_m.first().copied().unwrap_or(0.0);
        let c0 = params.initial;
        let inside = |v: usize| geo.tags[v] != ECS;
        let conc = [(c0.na_i, c0.na_e), (c0.k_i, c0.k_e), (c0.cl_i, c0.cl_e)]
            .map(|(ci, ce)| (0..n).map(|v| if inside(v) { ci } else { ce }).collect());
        let phi: Vec<f64> = (0..n)
            .map(|v| if inside(v) { phi_rest } else { 0.0 })
            .collect();

        // Membranes are closed to bulk diffusion; in the potential equation they conduct
        // only through the membrane current (set every step, see `solve_potential`).
        // Cell–cell contacts are closed.
        let h = geo.grid.h;
        let mut pot_faces: Vec<(Face, f64)> = faces.iter().map(|f| (f.face, 0.0)).collect();
        let contacts = geo.cell_contacts();
        pot_faces.extend(contacts.iter().map(|&f| (f, 0.0)));
        let closed: Vec<(Face, f64)> = pot_faces.iter().map(|&(f, _)| (f, 0.0)).collect();

        let bc = BoundaryConditions::neumann();
        let mut pot_op = DiffusionOperator::uniform(geo.grid, bc, 1.0);
        let coupling = vec![params.membrane.c_m / params.dt; faces.len()];
        let gamma = UA_PER_CM2_TO_FLUX * coupling[0];
        for (pf, f) in pot_faces.iter_mut().zip(&faces) {
            pf.1 = gamma * h[f.face.axis];
        }
        pot_op.set_face_sigma(&pot_faces);
        let conc_ops = SPECIES.map(|s| {
            let mut op = DiffusionOperator::uniform(geo.grid, bc, s.d);
            op.set_face_sigma(&closed);
            op.set_shift(&vec![1.0 / params.dt; n]);
            op
        });
        let conc_pre = [0, 1, 2].map(|k| Jacobi::new(&conc_ops[k]));
        let mut bulk_faces = Vec::new();
        let stride = [1, geo.grid.n[0], geo.grid.n[0] * geo.grid.n[1]];
        geo.grid.for_each_cell(|i, j, k, v| {
            for (a, pos) in [i, j, k].into_iter().enumerate() {
                let q = v + stride[a];
                if pos + 1 < geo.grid.n[a] && geo.tags[v] == geo.tags[q] {
                    bulk_faces.push((v as u32, q as u32, 1.0 / (h[a] * h[a])));
                }
            }
        });
        let mut solver = Self {
            pot_mg: CablePreconditioner::new(&pot_op, &geo),
            pot_op,
            conc_ops,
            conc_pre,
            pcg: Pcg::new(n),
            sigma: vec![0.0; n],
            rhs: vec![0.0; n],
            work: vec![0.0; n],
            delta: vec![0.0; n],
            face_work: vec![[0.0; 2]; bulk_faces.len()],
            pot_faces,
            coupling,
            bulk_faces,
            rho: vec![0.0; n],
            i_m: vec![0.0; faces.len()],
            psi: psi_mv(params.membrane.temperature),
            geo,
            params,
            faces,
            patches,
            conc,
            phi,
            time: 0.0,
            steps: 0,
            membrane_store: [0.0; 3],
            events: Vec::new(),
            stats: StepStats::default(),
        };
        solver.update_charge();
        solver
    }

    pub fn geometry(&self) -> &VoxelGeometry {
        &self.geo
    }

    pub fn params(&self) -> &EmiParams {
        &self.params
    }

    /// Membrane faces, in the same order as the patches.
    pub fn faces(&self) -> &[MembraneFace] {
        &self.faces
    }

    /// Concentration of species `k` (index into [`SPECIES`]) per voxel, mM.
    pub fn concentration(&self, k: usize) -> &[f64] {
        &self.conc[k]
    }

    /// Electric potential per voxel, mV.
    pub fn potential(&self) -> &[f64] {
        &self.phi
    }

    /// Membrane potential per face, mV.
    pub fn membrane_potential(&self) -> &[f64] {
        &self.patches.phi_m
    }

    /// Total membrane current density I_M of the last step per face, µA/cm² (outward).
    pub fn membrane_current(&self) -> &[f64] {
        &self.i_m
    }

    /// Charge density Σ_k z_k c_k per voxel, mM.
    pub fn charge_density(&self) -> &[f64] {
        &self.rho
    }

    pub fn last_stats(&self) -> StepStats {
        self.stats
    }

    /// Amount of each species in the domain, mM·µm³, counting the ions the capacitive
    /// currents moved onto the membrane. Constant in time with closed outer boundaries.
    pub fn total_ions(&self) -> [f64; 3] {
        let v = self.geo.voxel_volume();
        [0, 1, 2].map(|k| self.conc[k].iter().sum::<f64>() * v + self.membrane_store[k])
    }

    /// Opens synaptic Na⁺ channels with peak conductance `g` (mS/cm², decaying with the
    /// membrane's `syn_tau`) at time `t0` on the given patches. A patch holds one synapse
    /// at a time, so a later event on the same patch replaces the earlier one when it starts.
    pub fn add_synapse(&mut self, t0: f64, g: f64, patches: &[usize]) {
        assert!(
            patches.iter().all(|&p| p < self.faces.len()),
            "patch out of range"
        );
        let pos = self.events.partition_point(|e| e.t0 <= t0);
        let patches = patches.to_vec();
        self.events.insert(pos, SynapseEvent { t0, g, patches });
    }

    /// Patches of cell `cell` whose face centre lies at `x < x_max` (µm), e.g. one end of
    /// an axon for [`add_synapse`](Self::add_synapse).
    pub fn patches_near_start(&self, cell: u32, x_max: f64) -> Vec<usize> {
        (0..self.faces.len())
            .filter(|&i| {
                let f = &self.faces[i];
                f.cell == cell && f.center(&self.geo.grid)[0] < x_max
            })
            .collect()
    }

    /// Advances one PDE step Δt: membrane ODEs, then the potential, then the concentrations.
    pub fn step_once(&mut self) {
        let dt = self.params.dt;
        self.start_synapses(self.time + dt);
        self.load_patch_concentrations();
        // 1. ODE step with I_M = 0: φ_M → φ_M*, channel currents at φ_M* and old concentrations.
        self.patches
            .advance(self.time, dt, self.params.substeps, None);
        // 2. Potential with charge conservation and the capacitive membrane current.
        self.solve_potential();
        // 3. Concentrations: implicit diffusion, explicit drift, membrane ion fluxes.
        self.update_membrane();
        self.solve_concentrations();

        self.update_charge();
        self.time += dt;
        self.steps += 1;
    }

    fn start_synapses(&mut self, t_end: f64) {
        let due = self.events.partition_point(|e| e.t0 < t_end);
        for e in self.events.drain(..due) {
            for &p in &e.patches {
                self.patches.set_synapse(p, e.g, e.t0);
            }
        }
    }

    fn load_patch_concentrations(&mut self) {
        let p = &mut self.patches;
        let [na, k, cl] = &self.conc;
        for (i, f) in self.faces.iter().enumerate() {
            let (a, b) = (f.inside, f.outside);
            p.na_i[i] = na[a];
            p.na_e[i] = na[b];
            p.k_i[i] = k[a];
            p.k_e[i] = k[b];
            p.cl_i[i] = cl[a];
            p.cl_e[i] = cl[b];
        }
    }

    /// Conductivity σ = Σ z² D c / ψ per voxel, mM·µm²/(ms·mV).
    fn fill_conductivity(&mut self) {
        let psi = self.psi;
        let [c0, c1, c2] = &self.conc;
        for (v, s) in self.sigma.iter_mut().enumerate() {
            *s = SPECIES
                .iter()
                .zip([c0[v], c1[v], c2[v]])
                .map(|(sp, c)| (sp.z * sp.z) as f64 * sp.d * c)
                .sum::<f64>()
                / psi;
        }
    }

    /// Solves `-∇·(σ∇φ) = ρ/Δt + ∇·(Σ z D ∇c)` per voxel, with membrane faces carrying
    /// `I_M = w (φ_in − φ_out − φ_M*)` (see `coupling`).
    fn solve_potential(&mut self) {
        let dt = self.params.dt;
        self.fill_conductivity();
        // Membrane coupling: integrating C_M dφ_M/dt = I_M − g (φ_M − E) exactly over Δt
        // with I_M constant and g frozen gives I_M = (C_M/Δt) κ/(1 − e^{−κ}) (φ_M − φ_M*),
        // κ = gΔt/C_M, where φ_M* is the ODE result with I_M = 0. This keeps passive steady
        // states exact; for κ → 0 it is the paper's C_M (φ_M − φ_M*)/Δt.
        let c_dt = self.params.membrane.c_m / dt;
        let h = self.geo.grid.h;
        for ((w, pf), &g) in self
            .coupling
            .iter_mut()
            .zip(&mut self.pot_faces)
            .zip(&self.patches.g_ch)
        {
            let kappa = g / c_dt;
            let factor = if kappa < 1e-6 {
                1.0 + 0.5 * kappa
            } else {
                kappa / -(-kappa).exp_m1()
            };
            *w = c_dt * factor;
            pf.1 = UA_PER_CM2_TO_FLUX * *w * h[pf.0.axis];
        }
        self.pot_op.set_face_sigma(&self.pot_faces);
        self.pot_op.update_sigma(&self.sigma);
        // σ drifts slowly, so the preconditioner may lag the operator by a few steps.
        if self.steps % PRECONDITIONER_REFRESH == 0 {
            self.pot_mg.update(&self.pot_op);
        }

        // Diffusion currents: Σ z ∇·(D∇c) = -Σ z (A_k c - c/Δt) with the closed-membrane
        // concentration operators A_k = 1/Δt - ∇·D_k∇.
        for (v, b) in self.rhs.iter_mut().enumerate() {
            *b = self.rho[v] / dt;
        }
        for (k, sp) in SPECIES.iter().enumerate() {
            let z = sp.z as f64;
            self.conc_ops[k].apply(&self.conc[k], &mut self.work);
            for ((b, &ac), &c) in self.rhs.iter_mut().zip(&self.work).zip(&self.conc[k]) {
                *b -= z * (ac - c / dt);
            }
        }
        for ((f, &phi_star), &c) in self
            .faces
            .iter()
            .zip(&self.patches.phi_m)
            .zip(&self.coupling)
        {
            let w = UA_PER_CM2_TO_FLUX * c * phi_star / h[f.face.axis];
            self.rhs[f.inside] += w;
            self.rhs[f.outside] -= w;
        }

        // Solve for the increment δ in A (φ + δ) = b, so the tolerance is relative to this
        // step's change; any leftover residual carries into the next step's increment.
        self.pot_op.apply(&self.phi, &mut self.work);
        for (b, &a) in self.rhs.iter_mut().zip(&self.work) {
            *b -= a;
        }
        let opts = CgOptions {
            rtol: self.params.potential_rtol,
            null_space: NullSpace::Constant,
            max_iter: 500,
            ..Default::default()
        };
        self.delta.fill(0.0);
        self.stats.potential = self.pcg.solve(
            &self.pot_op,
            &mut self.pot_mg,
            &self.rhs,
            &mut self.delta,
            &opts,
        );
        for (p, d) in self.phi.iter_mut().zip(&self.delta) {
            *p += d;
        }
        // Fix the free constant: zero mean in the extracellular space.
        let (mut sum, mut count) = (0.0, 0usize);
        for (p, &t) in self.phi.iter().zip(&self.geo.tags) {
            if t == ECS {
                sum += p;
                count += 1;
            }
        }
        let mean = sum / count.max(1) as f64;
        self.phi.iter_mut().for_each(|p| *p -= mean);
    }

    /// φ_M = φ_in − φ_out and I_M from the membrane coupling on every face.
    fn update_membrane(&mut self) {
        for (i, f) in self.faces.iter().enumerate() {
            let phi_m = self.phi[f.inside] - self.phi[f.outside];
            self.i_m[i] = self.coupling[i] * (phi_m - self.patches.phi_m[i]);
            self.patches.phi_m[i] = phi_m;
        }
    }

    /// Fraction α_k = D_k z_k² c_k / Σ_l D_l z_l² c_l of the capacitive current carried by
    /// species `k` at voxel `v`.
    fn alpha(&self, k: usize, v: usize) -> f64 {
        let s = &SPECIES[k];
        (s.z * s.z) as f64 * s.d * self.conc[k][v] / (self.sigma[v] * self.psi)
    }

    fn solve_concentrations(&mut self) {
        let dt = self.params.dt;
        let g = self.geo.grid;
        let area = [0, 1, 2].map(|a| self.geo.face_area(a));
        // Per bulk face: σ-weights for the face concentration and -(φ_q − φ_v)/h².
        // Weighting c_face = (σ_q c_v + σ_v c_q)/(σ_v + σ_q) makes Σ_k z_k (drift flux)
        // equal the harmonic-mean conductivity used for φ.
        for (f, w) in self.bulk_faces.iter().zip(&mut self.face_work) {
            let (v, q) = (f.0 as usize, f.1 as usize);
            let (sv, sq) = (self.sigma[v], self.sigma[q]);
            let e = -(self.phi[q] - self.phi[v]) * f.2 / (sv + sq);
            *w = [sq * e, sv * e];
        }
        let opts = CgOptions {
            rtol: 1e-6,
            max_iter: 200,
            ..Default::default()
        };
        let mut max_it = 0;
        for (k, sp) in SPECIES.iter().enumerate() {
            let drift = sp.d * sp.z as f64 / self.psi;
            let c = &self.conc[k];
            // Increment form: A δ = c/Δt + sources − A c.
            self.conc_ops[k].apply(c, &mut self.rhs);
            for (b, &cv) in self.rhs.iter_mut().zip(c) {
                *b = cv / dt - *b;
            }
            // Explicit drift across bulk faces (flux from v to q per unit volume).
            for (f, w) in self.bulk_faces.iter().zip(&self.face_work) {
                let (v, q) = (f.0 as usize, f.1 as usize);
                let flux = drift * (w[0] * c[v] + w[1] * c[q]);
                self.rhs[v] -= flux;
                self.rhs[q] += flux;
            }
            // Membrane fluxes: F z J = I_ch^k + α_k (I_M − I_ch), α from each side.
            let i_ch_k = match k {
                0 => &self.patches.i_na,
                1 => &self.patches.i_k,
                _ => &self.patches.i_cl,
            };
            let to_flux = UA_PER_CM2_TO_FLUX / sp.z as f64;
            let mut store = 0.0;
            for (i, f) in self.faces.iter().enumerate() {
                let cap = self.i_m[i] - self.patches.i_ch[i];
                let out_in = to_flux * (i_ch_k[i] + self.alpha(k, f.inside) * cap);
                let out_ex = to_flux * (i_ch_k[i] + self.alpha(k, f.outside) * cap);
                let ah = 1.0 / g.h[f.face.axis];
                self.rhs[f.inside] -= out_in * ah;
                self.rhs[f.outside] += out_ex * ah;
                store += (out_in - out_ex) * area[f.face.axis];
            }
            self.membrane_store[k] += store * dt;
            self.delta.fill(0.0);
            let stats = self.pcg.solve(
                &self.conc_ops[k],
                &mut self.conc_pre[k],
                &self.rhs,
                &mut self.delta,
                &opts,
            );
            debug_assert!(stats.converged, "concentration solve: {stats:?}");
            max_it = max_it.max(stats.iterations);
            for (c, d) in self.conc[k].iter_mut().zip(&self.delta) {
                *c += d;
            }
        }
        self.stats.concentration_iterations = max_it;
    }

    fn update_charge(&mut self) {
        let [c0, c1, c2] = &self.conc;
        for (v, r) in self.rho.iter_mut().enumerate() {
            *r = SPECIES[0].z as f64 * c0[v]
                + SPECIES[1].z as f64 * c1[v]
                + SPECIES[2].z as f64 * c2[v];
        }
    }
}

impl Simulation for EmiSolver {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.step_once();
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}
