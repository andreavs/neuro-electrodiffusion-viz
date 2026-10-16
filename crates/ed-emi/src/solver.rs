//! The KNP-EMI time stepper: membrane ODE step, then explicit electroneutral PDE substeps.
//! The formulation is documented at the crate root.

use crate::geometry::{Geometry, MembraneFace, ECS};
use ed_core::constants::psi_mv;
use ed_core::ions::{CL, K, NA};
use ed_core::linalg::{
    BandCholesky, BoundaryConditions, CgOptions, CsrMatrix, DiffusionOperator, NullSpace, Pcg,
};
use ed_core::membrane::{IonConcentrations, MembraneParams, MembranePatches, PatchState};
use ed_core::Simulation;

/// Species order used by every per-species array: Na⁺, K⁺, Cl⁻.
pub const SPECIES: [ed_core::ions::IonSpecies; 3] = [NA, K, CL];

/// Current density (µA/cm²) carried by an ion flux of 1 mM·µm/ms of a monovalent ion:
/// `F · 1e-3 mol/(m² s)` = 96.48 A/m².
pub const CURRENT_PER_FLUX: f64 = ed_core::constants::F * 0.1;

/// Model and numerical parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmiParams {
    /// Time step Δt, ms.
    pub dt: f64,
    /// Membrane ODE substeps per time step.
    pub ode_substeps: usize,
    pub membrane: MembraneParams,
    /// Initial intra- and extracellular concentrations.
    pub concentrations: IonConcentrations,
    /// Initial membrane state; `None` starts at the membrane's resting state.
    pub initial_state: Option<PatchState>,
    /// Peak synaptic conductance used by [`KnpEmi::stimulate`], mS/cm².
    pub syn_g: f64,
    /// Length of the stimulated segment at the lower (`z`) end of an axon, µm.
    pub syn_length: f64,
    /// Relative tolerance of the potential solve.
    pub rtol: f64,
}

impl Default for EmiParams {
    fn default() -> Self {
        Self {
            dt: 0.1,
            ode_substeps: 25,
            membrane: MembraneParams::HODGKIN_HUXLEY,
            concentrations: IonConcentrations::KNP_EMI,
            initial_state: None,
            syn_g: 4.0,
            syn_length: 20.0,
            rtol: 1e-9,
        }
    }
}

/// Solver statistics from the most recent step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    /// PDE substeps per time step.
    pub substeps: usize,
    /// CG iterations summed over the substeps.
    pub cg_iterations: usize,
    /// Number of preconditioner factorisations so far.
    pub factorisations: usize,
}

/// CG iterations above which the preconditioner is refactored.
const REFACTOR_ITERATIONS: usize = 6;

/// Cell-scale KNP-EMI simulation on a voxel geometry.
pub struct KnpEmi {
    pub geometry: Geometry,
    pub params: EmiParams,
    time: f64,
    psi: f64,
    /// Concentrations (mM) per species and voxel.
    c: [Vec<f64>; 3],
    /// Electric potential (mV) per voxel, zero mean.
    phi: Vec<f64>,
    faces: Vec<MembraneFace>,
    /// Membrane state, one patch per face.
    pub patches: MembranePatches,
    /// Ions held in the membrane's capacitive layers per face, mM·µm (see crate docs).
    store: [Vec<f64>; 3],
    /// Whether each interior face (indexed by its lower voxel, per axis) joins two voxels
    /// of the same region.
    bulk: [Vec<bool>; 3],
    /// Per face, the fraction `(1 − e^{−κ})/κ`, `κ = g_ch Δt / C_M`, of a constant `I_M`
    /// that charges the capacitor over a step (the rest leaks through the channels).
    cap_frac: Vec<f64>,
    // Work buffers.
    i_mean: [Vec<f64>; 3],
    alpha: [[Vec<f64>; 3]; 2],
    kappa: Vec<f64>,
    face_sigma: [Vec<f64>; 3],
    rhs: Vec<f64>,
    /// Concentration increments of the current substep.
    dc: [Vec<f64>; 3],
    op: DiffusionOperator,
    chol: BandCholesky,
    pcg: Pcg,
    substeps: usize,
    stats: StepStats,
}

impl KnpEmi {
    pub fn new(geometry: Geometry, params: EmiParams) -> Self {
        assert!(params.dt > 0.0 && params.ode_substeps > 0);
        let g = geometry.grid;
        let n = g.len();
        let conc = params.concentrations;
        let mut c = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (v, &tag) in geometry.tags.iter().enumerate() {
            let (na, k, cl) = if tag == ECS {
                (conc.na_e, conc.k_e, conc.cl_e)
            } else {
                (conc.na_i, conc.k_i, conc.cl_i)
            };
            c[0][v] = na;
            c[1][v] = k;
            c[2][v] = cl;
        }
        let faces = geometry.membrane_faces();
        let patches = match params.initial_state {
            Some(s) => MembranePatches::with_state(faces.len(), params.membrane, conc, s),
            None => MembranePatches::new(faces.len(), params.membrane, conc),
        };
        let mut bulk = [vec![false; n], vec![false; n], vec![false; n]];
        geometry.for_each_face(|a, lo, hi| bulk[a][lo] = geometry.tags[lo] == geometry.tags[hi]);

        // Explicit stability limit of the concentration update (FTCS), with a margin for
        // migration.
        let d_max = SPECIES.iter().map(|s| s.d).fold(0.0, f64::max);
        let inv_h2: f64 = g.h.iter().map(|h| 1.0 / (h * h)).sum();
        let dt_max = 0.8 / (2.0 * d_max * inv_h2);
        let substeps = (params.dt / dt_max).ceil().max(1.0) as usize;

        let nf = faces.len();
        let zeros3 = || [vec![0.0; nf], vec![0.0; nf], vec![0.0; nf]];
        let op = DiffusionOperator::uniform(g, BoundaryConditions::neumann(), 0.0);
        let mut sim = Self {
            psi: psi_mv(params.membrane.temperature),
            time: 0.0,
            c,
            phi: vec![0.0; n],
            patches,
            store: zeros3(),
            i_mean: zeros3(),
            cap_frac: vec![1.0; nf],
            alpha: [zeros3(), zeros3()],
            bulk,
            kappa: vec![0.0; n],
            face_sigma: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            rhs: vec![0.0; n],
            dc: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            chol: BandCholesky::new(&CsrMatrix::from_triplets(0, 0, &[])),
            op,
            pcg: Pcg::new(n),
            substeps,
            stats: StepStats::default(),
            faces,
            geometry,
            params,
        };
        sim.assemble_operator();
        sim.refactor();
        sim
    }

    pub fn faces(&self) -> &[MembraneFace] {
        &self.faces
    }

    /// Concentrations (mM) of species `s` (index into [`SPECIES`]) per voxel.
    pub fn concentration(&self, s: usize) -> &[f64] {
        &self.c[s]
    }

    /// Potential (mV) per voxel. Only differences are meaningful; the mean is zero.
    pub fn potential(&self) -> &[f64] {
        &self.phi
    }

    /// Membrane potential φ_M (mV) per membrane face.
    pub fn membrane_potential(&self) -> &[f64] {
        &self.patches.phi_m
    }

    pub fn stats(&self) -> StepStats {
        self.stats
    }

    /// Total amount of species `s` in the system (mM·µm³ = 10⁻¹⁸ mol): bulk voxels plus the
    /// membrane's capacitive layers. Conserved exactly with closed outer boundaries.
    pub fn total_amount(&self, s: usize) -> f64 {
        let g = self.geometry.grid;
        let vol = g.h[0] * g.h[1] * g.h[2];
        let bulk: f64 = self.c[s].iter().sum::<f64>() * vol;
        let layer: f64 = self
            .faces
            .iter()
            .zip(&self.store[s])
            .map(|(f, q)| q * vol / g.h[f.axis as usize])
            .sum();
        bulk + layer
    }

    /// Opens synapses on the first [`EmiParams::syn_length`] µm (lowest `z`) of axon
    /// `cell` at time `t0` (ms), replacing any earlier synaptic input on those faces.
    pub fn stimulate(&mut self, cell: u16, t0: f64) {
        assert!(
            self.faces.iter().any(|f| f.cell == cell),
            "cell {cell} has no membrane"
        );
        let g = self.geometry.grid;
        let z0 = self
            .faces
            .iter()
            .filter(|f| f.cell == cell)
            .map(|f| self.z_of(f.inner))
            .fold(f64::INFINITY, f64::min);
        for (i, f) in self.faces.iter().enumerate() {
            if f.cell == cell && self.z_of(f.inner) - z0 < self.params.syn_length - 0.5 * g.h[2] {
                self.patches.set_synapse(i, self.params.syn_g, t0);
            }
        }
    }

    fn z_of(&self, voxel: usize) -> f64 {
        let g = self.geometry.grid;
        (voxel / (g.n[0] * g.n[1])) as f64 * g.h[2]
    }

    /// Face coefficients of the potential equation: mean conductivity on bulk faces,
    /// `C_M/Δt_s` (in flux units, times `h`) on membrane faces, zero elsewhere.
    fn assemble_operator(&mut self) {
        let psi = self.psi;
        for (v, k) in self.kappa.iter_mut().enumerate() {
            *k = SPECIES
                .iter()
                .zip(&self.c)
                .map(|(s, c)| s.d * (s.z * s.z) as f64 * c[v])
                .sum::<f64>()
                / psi;
        }
        let g = self.geometry.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        for ((fs, bulk), st) in self.face_sigma.iter_mut().zip(&self.bulk).zip(stride) {
            for (v, f) in fs.iter_mut().enumerate() {
                *f = if bulk[v] {
                    0.5 * (self.kappa[v] + self.kappa[v + st])
                } else {
                    0.0
                };
            }
        }
        for (i, f) in self.faces.iter().enumerate() {
            let a = f.axis as usize;
            self.face_sigma[a][f.inner.min(f.outer)] = self.membrane_conductance(i) * g.h[a];
        }
        let [x, y, z] = &self.face_sigma;
        self.op.set_face_sigma([x, y, z]);
    }

    /// Membrane coupling `C_M / (Δt_s · cap_frac)` of face `i`, expressed as an ion flux
    /// (mM·µm/ms) per mV.
    fn membrane_conductance(&self, i: usize) -> f64 {
        let dt_s = self.params.dt / self.substeps as f64;
        self.params.membrane.c_m / (dt_s * self.cap_frac[i]) / CURRENT_PER_FLUX
    }

    fn refactor(&mut self) {
        let a = self.op.to_csr();
        if self.chol.len() == a.n_rows() {
            self.chol.factor(&a);
        } else {
            self.chol = BandCholesky::new(&a);
        }
        self.stats.factorisations += 1;
    }

    /// Advances one time step Δt: membrane ODEs, then the PDE substeps.
    fn step_once(&mut self) {
        let dt = self.params.dt;
        self.load_patch_concentrations();
        let [m0, m1, m2] = &mut self.i_mean;
        self.patches.advance_with_mean_currents(
            self.time,
            dt,
            self.params.ode_substeps,
            None,
            [m0, m1, m2],
        );
        let k_dt = dt / self.params.membrane.c_m;
        for (f, &g) in self.cap_frac.iter_mut().zip(&self.patches.g_ch) {
            let kappa = g * k_dt;
            *f = if kappa > 1e-12 {
                -(-kappa).exp_m1() / kappa
            } else {
                1.0
            };
        }
        self.stats.cg_iterations = 0;
        self.stats.substeps = self.substeps;
        for _ in 0..self.substeps {
            self.pde_substep(dt / self.substeps as f64);
        }
        self.time += dt;
    }

    /// Copies voxel concentrations next to each face into the patches and computes the
    /// capacitive fractions α (eq. 12 of Publication 2) on both sides.
    fn load_patch_concentrations(&mut self) {
        let p = &mut self.patches;
        for (i, f) in self.faces.iter().enumerate() {
            let (vi, ve) = (f.inner, f.outer);
            p.na_i[i] = self.c[0][vi];
            p.k_i[i] = self.c[1][vi];
            p.cl_i[i] = self.c[2][vi];
            p.na_e[i] = self.c[0][ve];
            p.k_e[i] = self.c[1][ve];
            p.cl_e[i] = self.c[2][ve];
            for (side, v) in [vi, ve].into_iter().enumerate() {
                let w: [f64; 3] = std::array::from_fn(|s| {
                    SPECIES[s].d * (SPECIES[s].z.pow(2)) as f64 * self.c[s][v]
                });
                let total: f64 = w.iter().sum();
                debug_assert!(total > 0.0, "non-positive concentrations at voxel {v}");
                for (a, w) in self.alpha[side].iter_mut().zip(w) {
                    a[i] = w / total;
                }
            }
        }
    }

    fn pde_substep(&mut self, dt_s: f64) {
        let g = self.geometry.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        let inv_h2 = g.h.map(|h| 1.0 / (h * h));
        self.assemble_operator();

        // Right-hand side: minus the divergence of the diffusive charge flux, plus the
        // membrane terms G φ_M / h on both sides.
        self.rhs.fill(0.0);
        for a in 0..3 {
            for v in 0..g.len() {
                if !self.bulk[a][v] {
                    continue;
                }
                let nb = v + stride[a];
                let q: f64 = SPECIES
                    .iter()
                    .zip(&self.c)
                    .map(|(s, c)| s.z as f64 * s.d * (c[v] - c[nb]))
                    .sum::<f64>()
                    * inv_h2[a];
                self.rhs[v] -= q;
                self.rhs[nb] += q;
            }
        }
        for (i, (f, &phi_m)) in self.faces.iter().zip(&self.patches.phi_m).enumerate() {
            let m = self.membrane_conductance(i) * phi_m / g.h[f.axis as usize];
            self.rhs[f.inner] += m;
            self.rhs[f.outer] -= m;
        }

        let opts = CgOptions {
            rtol: self.params.rtol,
            atol: 1e-14,
            max_iter: 500,
            null_space: NullSpace::Constant,
        };
        let stats = self
            .pcg
            .solve(&self.op, &mut self.chol, &self.rhs, &mut self.phi, &opts);
        assert!(stats.converged, "potential solve failed: {stats:?}");
        self.stats.cg_iterations += stats.iterations;
        if stats.iterations > REFACTOR_ITERATIONS {
            self.refactor();
        }

        // Bulk Nernst-Planck fluxes with the new potential, from the concentrations at the
        // start of the substep (the same fluxes the potential equation balanced).
        let psi = self.psi;
        self.dc.iter_mut().for_each(|d| d.fill(0.0));
        for a in 0..3 {
            let k_h = dt_s * inv_h2[a];
            for v in 0..g.len() {
                if !self.bulk[a][v] {
                    continue;
                }
                let nb = v + stride[a];
                let dphi = self.phi[v] - self.phi[nb];
                for ((s, c), d) in SPECIES.iter().zip(&self.c).zip(&mut self.dc) {
                    let c_mid = 0.5 * (c[v] + c[nb]);
                    let flux = k_h * s.d * (c[v] - c[nb] + s.z as f64 * c_mid * dphi / psi);
                    d[v] -= flux;
                    d[nb] += flux;
                }
            }
        }

        // Membrane: total current I_M from the potential jump, then ion-specific fluxes
        // (eq. 13 of Publication 2) and the membrane potential update.
        let c_m = self.params.membrane.c_m;
        for (i, f) in self.faces.iter().enumerate() {
            let g_mem = self.membrane_conductance(i);
            let phi_m = &mut self.patches.phi_m[i];
            let i_m = g_mem * CURRENT_PER_FLUX * (self.phi[f.inner] - self.phi[f.outer] - *phi_m);
            *phi_m += self.cap_frac[i] * dt_s * i_m / c_m;
            let i_ch: f64 = self.i_mean.iter().map(|m| m[i]).sum();
            let k_h = dt_s / g.h[f.axis as usize];
            for (s, species) in SPECIES.iter().enumerate() {
                let zf = species.z as f64 * CURRENT_PER_FLUX;
                let i_k = self.i_mean[s][i];
                let into_i = -(i_k + self.alpha[0][s][i] * (i_m - i_ch)) / zf;
                let into_e = (i_k + self.alpha[1][s][i] * (i_m - i_ch)) / zf;
                self.dc[s][f.inner] += k_h * into_i;
                self.dc[s][f.outer] += k_h * into_e;
                self.store[s][i] -= dt_s * (into_i + into_e);
            }
        }
        for (c, d) in self.c.iter_mut().zip(&self.dc) {
            c.iter_mut().zip(d).for_each(|(c, d)| *c += d);
        }
    }
}

impl Simulation for KnpEmi {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.step_once();
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}
