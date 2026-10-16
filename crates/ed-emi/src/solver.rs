//! KNP-EMI time stepping: membrane ODE step, then a backward Euler PDE step.
//!
//! Units: µm, ms, mM, mV; membrane currents µA/cm², capacitance µF/cm². Ion fluxes are in
//! mM·µm/ms, and a flux `J` of an ion with valence `z` carries `F_U z J` µA/cm².
//!
//! One step from `t` to `t + Δt` (see docs/ARCHITECTURE.md, scale 2):
//! 1. HH membranes: advance the gates and φ_M with `I_M = 0` over `ode_substeps` substeps,
//!    giving φ_M* and the per-ion channel currents averaged exactly over the step (the paper
//!    evaluates them at φ_M*, eq. 31; the mean moves exactly the charge that left the
//!    capacitor).
//!    Passive membranes skip this; their leak currents are implicit in step 2 (paper eq. 30).
//! 2. Solve the charge equations for all potentials at once (backward Euler). Eliminating the
//!    lumped interface current, `I_M = C_M (φ_i − φ_e − φ_M*) / Δt` for HH or
//!    `C_M (φ_i − φ_e − φ_M^{n−1}) / Δt + Σ_k g_k (φ_i − φ_e − E_k)` for passive membranes,
//!    couples the two DOFs of each membrane vertex, giving one SPD system.
//! 3. Solve one implicit diffusion system per species with the paper's linearised flux
//!    `J = −D∇[k]^n − (D z/ψ)[k]^{n−1}∇φ^n` and the membrane fluxes of eq. 13.
//!
//! The linear solves are warm-started increments with a loose relative tolerance
//! ([`EmiParams::rtol`]); a per-subdomain correction keeps every ion amount exactly
//! consistent with the membrane fluxes, and the charge-correction term in step 2 keeps the
//! bulk electroneutral.

use ed_core::constants::{psi_mv, F};
use ed_core::ions::{IonSpecies, CL, K, NA};
use ed_core::linalg::{CgOptions, CgStats, CsrMatrix, NullSpace, Pcg, TwoLevel};
use ed_core::membrane::{IonConcentrations, MembraneParams, MembranePatches};

use crate::fem::{aggregates, solve_increment, Stiffness};
use crate::mesh::Mesh;

/// µA/cm² carried by a flux of 1 mM·µm/ms of a monovalent ion.
pub const F_U: f64 = F * 1e-1;

/// Steps between refreshes of the potential preconditioner.
const PC_REFRESH: u64 = 20;

/// The three species of Publication 2.
pub const SPECIES: [IonSpecies; 3] = [NA, K, CL];

/// Model and solver parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct EmiParams {
    /// PDE time step Δt, ms.
    pub dt: f64,
    /// Membrane ODE substeps per Δt (HH only). The paper used 25 forward Euler substeps; the
    /// exponential φ_M update and Rush-Larsen gates used here are stable for any substep,
    /// and 5 changes the conduction velocity by ~3% relative to 25.
    pub ode_substeps: usize,
    pub membrane: MembraneParams,
    /// Initial intra- and extracellular concentrations, mM (bulk electroneutral).
    pub initial: IonConcentrations,
    /// Relative tolerance of the linear solves, relative to each step's change.
    pub rtol: f64,
}

impl Default for EmiParams {
    fn default() -> Self {
        Self {
            dt: 0.1,
            ode_substeps: 5,
            membrane: MembraneParams::HODGKIN_HUXLEY,
            initial: IonConcentrations::KNP_EMI,
            rtol: 1e-2,
        }
    }
}

/// Why [`EmiSolver::step`] failed. The state is then unusable; build a new solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepError {
    /// A linear solve hit its iteration limit.
    NotConverged,
    /// A potential or concentration became non-finite, or a concentration non-positive
    /// (typically a time step too large for the membrane currents).
    InvalidState,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotConverged => "linear solve did not converge",
            Self::InvalidState => "non-finite potential or non-positive concentration",
        })
    }
}

impl std::error::Error for StepError {}

/// Iteration counts of the last step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepStats {
    pub potential: Option<CgStats>,
    pub concentration: [Option<CgStats>; 3],
}

/// The KNP-EMI solver state. Read it through the getters; the fields are private so that
/// the cached matrices always match the mesh and parameters.
pub struct EmiSolver {
    mesh: Mesh,
    params: EmiParams,
    t: f64,
    steps: u64,
    phi: Vec<f64>,
    conc: [Vec<f64>; 3],
    patches: MembranePatches,
    i_m: Vec<f64>,
    i_ch_mean: [Vec<f64>; 3],
    stats: StepStats,
    /// Potential CG iterations just after the last preconditioner refresh.
    pc_iterations: usize,
    /// Refresh the potential preconditioner on the next step.
    pc_stale: bool,
    psi: f64,
    stiffness: Stiffness,
    a_phi: CsrMatrix,
    pc_phi: TwoLevel,
    a_conc: [CsrMatrix; 3],
    pc_conc: [TwoLevel; 3],
    cg: Pcg,
    // Scratch, reused every step.
    sigma: Vec<f64>,
    cbar: [Vec<f64>; 3],
    domain_volume: Vec<f64>,
    weight: Vec<f64>,
    coupling: Vec<f64>,
    mem_src: Vec<f64>,
    g_impl: [Vec<f64>; 3],
    e_rev: [Vec<f64>; 3],
    rhs: [Vec<f64>; 3],
    work: [Vec<f64>; 2],
    defect: Vec<f64>,
}

/// Tolerances below ~1e-10 stall on roundoff in the constant-null-space potential solve;
/// `rtol ≥ 1` would skip every solve.
fn valid_accuracy(rtol: f64, ode_substeps: usize) -> bool {
    (1e-10..1.0).contains(&rtol) && ode_substeps > 0
}

impl EmiSolver {
    /// A solver at rest: uniform initial concentrations, φ_e = 0, φ_i = φ_M at the membrane
    /// rest state for those concentrations. Panics unless `dt` and `ode_substeps` are
    /// positive, `1e-10 ≤ rtol < 1`, and the mesh has extracellular space.
    pub fn new(mesh: Mesh, params: EmiParams) -> Self {
        assert!(
            params.dt > 0.0 && params.dt.is_finite(),
            "dt must be positive"
        );
        assert!(
            valid_accuracy(params.rtol, params.ode_substeps),
            "need 1e-10 ≤ rtol < 1 and substeps > 0"
        );
        assert!(
            mesh.dof_tag.contains(&0),
            "the mesh needs extracellular space"
        );
        let n = mesh.n_dofs();
        let c0 = params.initial;
        let patches = MembranePatches::new(mesh.membrane.len(), params.membrane, c0);
        let phi_rest = patches.phi_m.first().copied().unwrap_or(0.0);
        let phi = (0..n)
            .map(|d| if mesh.dof_tag[d] == 0 { 0.0 } else { phi_rest })
            .collect();
        let init = |i: f64, e: f64| -> Vec<f64> {
            (0..n)
                .map(|d| if mesh.dof_tag[d] == 0 { e } else { i })
                .collect()
        };
        let conc = [
            init(c0.na_i, c0.na_e),
            init(c0.k_i, c0.k_e),
            init(c0.cl_i, c0.cl_e),
        ];

        let mut domain_volume = vec![0.0; mesh.n_cells + 1];
        for d in 0..n {
            domain_volume[mesh.dof_tag[d] as usize] += mesh.lumped_volume[d];
        }
        let stiffness = Stiffness::new(&mesh);
        let n_tets = mesh.tets.len();
        let agg = aggregates(&mesh);
        let mut a_phi = stiffness.new_matrix();
        stiffness.assemble(
            &vec![1.0; n_tets],
            &vec![0.0; n],
            &vec![1.0; mesh.membrane.len()],
            &mut a_phi,
        );
        let pc_phi = TwoLevel::new(&a_phi, agg.clone());
        let shift: Vec<f64> = mesh.lumped_volume.iter().map(|v| v / params.dt).collect();
        let no_coupling = vec![0.0; mesh.membrane.len()];
        let a_conc = SPECIES.map(|s| {
            let mut a = stiffness.new_matrix();
            stiffness.assemble(&vec![s.d; n_tets], &shift, &no_coupling, &mut a);
            a
        });
        let pc_conc = [0, 1, 2].map(|k| TwoLevel::new(&a_conc[k], agg.clone()));

        Self {
            psi: psi_mv(params.membrane.temperature),
            i_m: vec![0.0; mesh.membrane.len()],
            i_ch_mean: [0, 1, 2].map(|_| vec![0.0; mesh.membrane.len()]),
            stats: StepStats::default(),
            pc_iterations: 0,
            pc_stale: false,
            t: 0.0,
            steps: 0,
            phi,
            conc,
            patches,
            stiffness,
            a_phi,
            pc_phi,
            a_conc,
            pc_conc,
            cg: Pcg::new(n),
            sigma: vec![0.0; n_tets],
            weight: vec![0.0; n_tets],
            coupling: vec![0.0; mesh.membrane.len()],
            mem_src: vec![0.0; mesh.membrane.len()],
            g_impl: [0, 1, 2].map(|_| vec![0.0; mesh.membrane.len()]),
            e_rev: [0, 1, 2].map(|_| vec![0.0; mesh.membrane.len()]),
            rhs: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            cbar: [vec![0.0; n_tets], vec![0.0; n_tets], vec![0.0; n_tets]],
            domain_volume,
            work: [vec![0.0; n], vec![0.0; n]],
            defect: vec![0.0; mesh.n_cells + 1],
            mesh,
            params,
        }
    }

    /// Opens a synapse (`g` mS/cm², onset `t0` ms) on the membrane of `cell` (tag ≥ 1) where
    /// `x_lo ≤ x ≤ x_hi`.
    pub fn stimulate(&mut self, cell: u16, x_lo: f64, x_hi: f64, g: f64, t0: f64) {
        self.stimulate_where(cell, |p| p[0] >= x_lo && p[0] <= x_hi, g, t0);
    }

    /// Opens a synapse (`g` mS/cm², onset `t0` ms) on every membrane node of `cell` (tag ≥ 1)
    /// whose position (µm) satisfies `inside`. Returns how many nodes it opened.
    pub fn stimulate_where(
        &mut self,
        cell: u16,
        inside: impl Fn([f64; 3]) -> bool,
        g: f64,
        t0: f64,
    ) -> usize {
        let mut n = 0;
        for (p, m) in self.mesh.membrane.iter().enumerate() {
            if m.cell == cell && inside(self.mesh.dof_position(m.inner as usize)) {
                self.patches.set_synapse(p, g, t0);
                n += 1;
            }
        }
        n
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn params(&self) -> &EmiParams {
        &self.params
    }

    /// Time, ms.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// Potential per DOF, mV. Gauged so the volume mean of φ_e is 0.
    pub fn phi(&self) -> &[f64] {
        &self.phi
    }

    /// Concentration of species `k` (Na⁺, K⁺, Cl⁻) per DOF, mM.
    pub fn conc(&self, k: usize) -> &[f64] {
        &self.conc[k]
    }

    /// Membrane potential φ_i − φ_e per membrane node (order of `mesh().membrane`), mV.
    pub fn phi_m(&self) -> &[f64] {
        &self.patches.phi_m
    }

    /// Membrane state (gates, synapses). Its `i_*` current fields are not kept in step with
    /// the PDE step; use [`Self::i_m`] and [`Self::i_ch_mean`].
    pub fn patches(&self) -> &MembranePatches {
        &self.patches
    }

    /// Total membrane current I_M per membrane node over the last step, µA/cm², outward.
    pub fn i_m(&self) -> &[f64] {
        &self.i_m
    }

    /// Channel current of species `k` per membrane node, averaged over the last step,
    /// µA/cm², outward. These are the currents that move ions in step 3.
    pub fn i_ch_mean(&self, k: usize) -> &[f64] {
        &self.i_ch_mean[k]
    }

    /// Iteration counts of the last step.
    pub fn stats(&self) -> StepStats {
        self.stats
    }

    /// Changes Δt, reassembling the concentration matrices. Panics unless `dt > 0`.
    pub fn set_dt(&mut self, dt: f64) {
        assert!(dt > 0.0 && dt.is_finite(), "dt must be positive");
        self.params.dt = dt;
        let n_tets = self.mesh.tets.len();
        let shift: Vec<f64> = self.mesh.lumped_volume.iter().map(|v| v / dt).collect();
        let no_coupling = vec![0.0; self.mesh.membrane.len()];
        for (k, sp) in SPECIES.iter().enumerate() {
            let a = &mut self.a_conc[k];
            self.stiffness
                .assemble(&vec![sp.d; n_tets], &shift, &no_coupling, a);
            self.pc_conc[k].update(a);
        }
        // The membrane coupling C_M/Δt in the potential matrix changes too.
        self.pc_stale = true;
    }

    /// Changes the linear solver tolerance and the ODE substeps. Panics unless
    /// `1e-10 ≤ rtol < 1` and `ode_substeps > 0`.
    pub fn set_accuracy(&mut self, rtol: f64, ode_substeps: usize) {
        assert!(
            valid_accuracy(rtol, ode_substeps),
            "need 1e-10 ≤ rtol < 1 and substeps > 0"
        );
        self.params.rtol = rtol;
        self.params.ode_substeps = ode_substeps;
    }

    /// Advances the solution by one time step Δt.
    pub fn step(&mut self) -> Result<(), StepError> {
        let dt = self.params.dt;
        let opts = CgOptions {
            rtol: self.params.rtol,
            atol: 1e-14,
            max_iter: 2000,
            null_space: NullSpace::None,
        };

        // 1. Membrane ODE step with the current concentrations.
        for (p, m) in self.mesh.membrane.iter().enumerate() {
            let (i, o) = (m.inner as usize, m.outer as usize);
            self.patches.na_i[p] = self.conc[0][i];
            self.patches.na_e[p] = self.conc[0][o];
            self.patches.k_i[p] = self.conc[1][i];
            self.patches.k_e[p] = self.conc[1][o];
            self.patches.cl_i[p] = self.conc[2][i];
            self.patches.cl_e[p] = self.conc[2][o];
        }
        // HH: split as in the paper. Passive: no ODE step; the leak currents enter the PDE step
        // implicitly (paper eq. 30), which avoids the O(Δt) splitting error in φ_M.
        let passive = !self.params.membrane.hodgkin_huxley;
        if !passive {
            let [na, k, cl] = &mut self.i_ch_mean;
            self.patches.advance_with_mean_currents(
                self.t,
                dt,
                self.params.ode_substeps,
                None,
                [na, k, cl],
            );
        }

        // 2. Potentials. Element means of [k]^{n−1} give σ and the drift weights.
        let c_m = self.params.membrane.c_m;
        for (cbar, c) in self.cbar.iter_mut().zip(&self.conc) {
            for (t, tet) in self.mesh.tets.iter().enumerate() {
                cbar[t] = 0.25 * tet.iter().map(|&d| c[d as usize]).sum::<f64>();
            }
        }
        for (t, s) in self.sigma.iter_mut().enumerate() {
            *s = SPECIES
                .iter()
                .zip(&self.cbar)
                .map(|(sp, cbar)| sp.d * (sp.z * sp.z) as f64 * cbar[t])
                .sum::<f64>()
                / self.psi;
        }
        // Per membrane node, I_M = (C_M/Δt + g)(φ_i − φ_e) − src, with g the implicit channel
        // conductance (0 for HH, whose channel currents were applied in the ODE step).
        let psi = self.psi;
        let t_new = self.t + dt;
        for (p, m) in self.mesh.membrane.iter().enumerate() {
            let phi_old = self.patches.phi_m[p];
            let (g, src) = if passive {
                let (i, o) = (m.inner as usize, m.outer as usize);
                let mp = &self.params.membrane;
                let g = [
                    mp.g_leak_na + self.patches.synaptic_conductance(p, t_new),
                    mp.g_leak_k,
                    mp.g_leak_cl,
                ];
                let mut src = c_m / dt * phi_old;
                for k in 0..3 {
                    let z = SPECIES[k].z as f64;
                    self.e_rev[k][p] = psi / z * (self.conc[k][o] / self.conc[k][i]).ln();
                    self.g_impl[k][p] = g[k];
                    src += g[k] * self.e_rev[k][p];
                }
                (g.iter().sum::<f64>(), src)
            } else {
                (0.0, c_m / dt * phi_old)
            };
            self.coupling[p] = m.area * (c_m / dt + g) / F_U;
            self.mem_src[p] = src;
        }
        let [rhs, diffusive, _] = &mut self.rhs;
        rhs.fill(0.0);
        self.stiffness
            .assemble(&self.sigma, rhs, &self.coupling, &mut self.a_phi);
        // σ drifts slowly with the concentrations, so the preconditioner is refreshed only
        // now and then, or when the iteration count has doubled (a synapse opening); PCG
        // stays exact for any fixed SPD preconditioner.
        let stale = self
            .stats
            .potential
            .is_some_and(|s| s.iterations > 2 * self.pc_iterations + 5);
        let refresh = self.steps % PC_REFRESH == 0 || stale || self.pc_stale;
        self.pc_stale = false;
        if refresh {
            self.pc_phi.update(&self.a_phi);
        }

        // Charge correction M Σ z [k] / Δt and the diffusion current −K Σ z D [k]. The latter
        // uses [k]^{n−1} (step 3 uses [k]^n); the charge correction absorbs the difference on
        // the next step.
        for d in 0..rhs.len() {
            let (mut charge, mut zd) = (0.0, 0.0);
            for (sp, c) in SPECIES.iter().zip(&self.conc) {
                charge += sp.z as f64 * c[d];
                zd -= (sp.z as f64) * sp.d * c[d];
            }
            rhs[d] = self.mesh.lumped_volume[d] * charge / dt;
            diffusive[d] = zd;
        }
        self.weight.fill(1.0);
        self.stiffness.apply_add(&self.weight, diffusive, rhs);
        for (p, m) in self.mesh.membrane.iter().enumerate() {
            let f = m.area * self.mem_src[p] / F_U;
            rhs[m.inner as usize] += f;
            rhs[m.outer as usize] -= f;
        }
        let stats = solve_increment(
            &mut self.cg,
            &self.a_phi,
            &mut self.pc_phi,
            rhs,
            &mut self.phi,
            &mut self.work,
            &CgOptions {
                null_space: NullSpace::Constant,
                ..opts
            },
        );
        self.stats.potential = Some(stats);
        if refresh {
            self.pc_iterations = stats.iterations;
        }
        // Only jumps and gradients of φ matter; fix the gauge at mean φ_e = 0.
        let (mut sum, ecs) = (0.0, self.domain_volume[0]);
        for d in 0..self.phi.len() {
            if self.mesh.dof_tag[d] == 0 {
                sum += self.mesh.lumped_volume[d] * self.phi[d];
            }
        }
        self.phi.iter_mut().for_each(|p| *p -= sum / ecs);

        for (p, m) in self.mesh.membrane.iter().enumerate() {
            let jump = self.phi[m.inner as usize] - self.phi[m.outer as usize];
            self.i_m[p] = self.coupling[p] * F_U / m.area * jump - self.mem_src[p];
            if passive {
                for k in 0..3 {
                    self.i_ch_mean[k][p] = self.g_impl[k][p] * (jump - self.e_rev[k][p]);
                }
            }
        }

        // 3. Concentrations. Right-hand sides first, while every [k] still holds step n−1.
        for (k, sp) in SPECIES.iter().enumerate() {
            let z = sp.z as f64;
            let rhs = &mut self.rhs[k];
            for ((r, v), c) in rhs
                .iter_mut()
                .zip(&self.mesh.lumped_volume)
                .zip(&self.conc[k])
            {
                *r = v * c / dt;
            }
            for (w, cbar) in self.weight.iter_mut().zip(&self.cbar[k]) {
                *w = -sp.d * z / self.psi * cbar;
            }
            self.stiffness.apply_add(&self.weight, &self.phi, rhs);
        }
        let dz2c = |d: usize, k: usize| {
            let sp = SPECIES[k];
            sp.d * (sp.z * sp.z) as f64 * self.conc[k][d]
        };
        for (p, m) in self.mesh.membrane.iter().enumerate() {
            let (i, o) = (m.inner as usize, m.outer as usize);
            let (den_i, den_e) = (
                (0..3).map(|k| dz2c(i, k)).sum::<f64>(),
                (0..3).map(|k| dz2c(o, k)).sum::<f64>(),
            );
            let i_ch = [0, 1, 2].map(|k| self.i_ch_mean[k][p]);
            // The capacitive current C_M (φ_M^n − φ_M^{n−1}) / Δt, split by α_r^k (eq. 13).
            let i_cap = self.i_m[p] - i_ch.iter().sum::<f64>();
            for k in 0..3 {
                let scale = m.area / (F_U * SPECIES[k].z as f64);
                self.rhs[k][i] -= scale * (i_ch[k] + dz2c(i, k) / den_i * i_cap);
                self.rhs[k][o] += scale * (i_ch[k] + dz2c(o, k) / den_e * i_cap);
            }
        }
        for k in 0..3 {
            let stats = solve_increment(
                &mut self.cg,
                &self.a_conc[k],
                &mut self.pc_conc[k],
                &self.rhs[k],
                &mut self.conc[k],
                &mut self.work,
                &opts,
            );
            self.stats.concentration[k] = Some(stats);
            // Exact conservation despite the loose tolerance: the stiffness has zero row sums
            // within each subdomain, so Σ_T M [k]^n must equal Δt Σ_T rhs on every subdomain T.
            let defect = &mut self.defect;
            defect.fill(0.0);
            for d in 0..self.mesh.n_dofs() {
                defect[self.mesh.dof_tag[d] as usize] +=
                    dt * self.rhs[k][d] - self.mesh.lumped_volume[d] * self.conc[k][d];
            }
            for d in 0..self.mesh.n_dofs() {
                let tag = self.mesh.dof_tag[d] as usize;
                self.conc[k][d] += defect[tag] / self.domain_volume[tag];
            }
        }

        for (p, m) in self.mesh.membrane.iter().enumerate() {
            self.patches.phi_m[p] = self.phi[m.inner as usize] - self.phi[m.outer as usize];
        }
        self.t += dt;
        self.steps += 1;

        // A non-finite state also breaks CG, so check it first for the more useful error.
        let positive = self
            .conc
            .iter()
            .flatten()
            .all(|&c| c > 0.0 && c.is_finite());
        if !positive || !self.phi.iter().all(|p| p.is_finite()) {
            return Err(StepError::InvalidState);
        }
        let stats = self
            .stats
            .concentration
            .iter()
            .chain([&self.stats.potential]);
        if !stats.flatten().all(|s| s.converged) {
            return Err(StepError::NotConverged);
        }
        Ok(())
    }

    /// Total amount of species `k` in subdomain `tag` (lumped), mM·µm³.
    pub fn amount(&self, k: usize, tag: u16) -> f64 {
        (0..self.mesh.n_dofs())
            .filter(|&d| self.mesh.dof_tag[d] == tag)
            .map(|d| self.mesh.lumped_volume[d] * self.conc[k][d])
            .sum()
    }

    /// Largest nodal |Σ_k z_k [k]| over all DOFs, mM.
    pub fn max_charge_imbalance(&self) -> f64 {
        (0..self.mesh.n_dofs())
            .map(|d| {
                SPECIES
                    .iter()
                    .zip(&self.conc)
                    .map(|(sp, c)| sp.z as f64 * c[d])
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Aabb, MeshSpec};

    fn axon(params: EmiParams) -> EmiSolver {
        let spec = MeshSpec {
            extent: [60.0, 0.8, 0.8],
            cells: vec![Aabb::new([5.0, 0.3, 0.3], [55.0, 0.5, 0.5])],
            max_h: [5.0, 0.3, 0.3],
        };
        EmiSolver::new(Mesh::build(&spec).unwrap(), params)
    }

    #[test]
    fn axon_at_rest_stays_at_rest_and_neutral_after_a_dt_change() {
        let mut s = axon(EmiParams::default());
        let rest = s.phi_m()[0];
        for i in 0..100 {
            if i == 50 {
                s.set_dt(0.05);
            }
            s.step().unwrap();
        }
        let drift = s.phi_m().iter().map(|v| (v - rest).abs());
        assert!(drift.fold(0.0, f64::max) < 0.5, "φ_M left rest {rest} mV");
        assert!(s.max_charge_imbalance() < 1e-3);
        assert!((s.t() - 7.5).abs() < 1e-9);
    }

    #[test]
    fn extracellular_potential_has_zero_mean() {
        let mut s = axon(EmiParams::default());
        s.stimulate(1, 0.0, 20.0, 4.0, 0.0);
        for _ in 0..20 {
            s.step().unwrap();
        }
        let m = s.mesh();
        let mean: f64 = (0..m.n_dofs())
            .filter(|&d| m.dof_tag[d] == 0)
            .map(|d| m.lumped_volume[d] * s.phi()[d])
            .sum();
        assert!(mean.abs() < 1e-9 * s.domain_volume[0]);
    }

    #[test]
    fn an_impossible_step_is_reported() {
        let mut membrane = MembraneParams::HODGKIN_HUXLEY;
        membrane.hodgkin_huxley = false;
        let mut s = axon(EmiParams {
            membrane,
            ..EmiParams::default()
        });
        s.stimulate(1, 0.0, 60.0, 1e5, 0.0);
        let failed = (0..20).map(|_| s.step()).find_map(Result::err);
        assert!(failed.is_some());
    }
}
