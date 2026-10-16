use crate::flux::{DriftFactors, FaceFluxes};
use crate::{Ion, KnpParams, OuterBoundary, NA_TO_MM_UM3_PER_MS};
use ed_core::constants::psi_mv;
use ed_core::linalg::{
    BoundaryConditions, CgOptions, DiffusionOperator, Grid, LinearOperator, Multigrid, NullSpace,
    Pcg,
};
use ed_core::Simulation;

/// Solver work from the last call to [`KnpSolver::step`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    /// Explicit substeps the time step was split into.
    pub substeps: usize,
    /// CG iterations of the last potential solve.
    pub cg_iterations: usize,
    /// Relative residual of the last potential solve.
    pub cg_residual: f64,
    /// Whether the last potential solve reached its tolerance (also true when it was
    /// skipped because no species carries charge).
    pub converged: bool,
}

/// Explicit finite-volume KNP solver on a uniform 3-D grid. Fields are flat arrays in
/// [`Grid`] order (x fastest).
///
/// Each substep solves the potential equation for the current concentrations, then
/// moves every species with forward Euler using face fluxes. Diffusion and drift are
/// explicit because the diffusive stability limit (about 34 ms on the default grid) is
/// far above the 0.1 ms steps that neuron sources need; longer steps are split into
/// substeps. The central drift flux also needs potential jumps below about 2ψ/|z|
/// (≈ 50 mV for z = 1) between neighbouring cells, and concentrations must stay
/// positive; both hold by orders of magnitude for neuronal sources.
///
/// The drift flux of each species uses the arithmetic mean concentration on a face,
/// scaled by `σ_op,harmonic / σ_arithmetic`, where `σ_op` is the conductivity of the
/// potential operator. The charge flux `Σ z J` then matches that operator exactly.
/// Every species is conserved to round-off, and `Σ z c` changes only by the capacitive
/// source plus the CG residual, at most [`CHARGE_TOL`] per cell. That error does not
/// accumulate: the deviation of `Σ z c` from its expected value is added to the next
/// potential solve, which removes it.
///
/// CG starts from a linear extrapolation of the last two potentials, and the operator
/// with its multigrid preconditioner is refreshed only every few substeps, so a typical
/// step costs one V-cycle or less.
pub struct KnpSolver {
    grid: Grid,
    ions: Vec<Ion>,
    /// Effective diffusion coefficients D̃ (µm²/ms) and valences.
    d: Vec<f64>,
    z: Vec<f64>,
    psi: f64,
    alpha: f64,
    dt: f64,
    dt_max: f64,
    time: f64,
    c: Vec<Vec<f64>>,
    c_next: Vec<f64>,
    phi: Vec<f64>,
    /// Potential of the previous substep, for a linear-extrapolation CG start.
    phi_prev: Vec<f64>,
    /// Ion sources (mM/ms of ECS), the capacitive source and the total charge source
    /// `Σ z f̃ + ĩ_cap/F` (mM/ms).
    src: Vec<Vec<f64>>,
    cap: Vec<f64>,
    charge_src: Vec<f64>,
    /// Cells holding a source, so clearing touches only those.
    source_cells: Vec<usize>,
    is_source: Vec<bool>,
    /// `Σ z c`, and the value it should have: its value when the concentrations were
    /// last set, minus the integrated capacitive source. `None` until the next substep.
    charge: Vec<f64>,
    charge_ref: Option<Vec<f64>>,
    /// Conductivity / F, `Σ z D̃ (c − c_baseline)`, and the potential RHS.
    sigma: Vec<f64>,
    zdc: Vec<f64>,
    rhs: Vec<f64>,
    drift: DriftFactors,
    /// `−Δ` with unit coefficient, and its action on constants: `Σ 2/h²` over the
    /// Dirichlet faces of each cell (all zero for no-flux boundaries).
    laplace: DiffusionOperator,
    ghost: Vec<f64>,
    op: DiffusionOperator,
    mg: Multigrid,
    /// Substeps since `op` and `mg` were last refreshed.
    mg_age: usize,
    pcg: Pcg,
    cg_opts: CgOptions,
    stats: StepStats,
}

/// Refresh the potential operator and its multigrid hierarchy from the current
/// conductivity every this many substeps. In between, the drift fluxes use the
/// operator's face conductivities, so charge stays exactly consistent; the lag only
/// delays how conductivity changes (a fraction of a percent per second) affect φ.
const OPERATOR_REFRESH_INTERVAL: usize = 16;

/// Largest charge density error (mM) a potential solve may leave in a cell: about 10⁻¹⁰
/// of the ion concentrations, and below the physical charge density that sustains
/// diffusion potentials (~10⁻⁹ of them, Publication 1). The next solve removes it.
pub const CHARGE_TOL: f64 = 1e-8;

impl KnpSolver {
    /// Builds a solver with every species at its baseline concentration and `φ = 0`.
    pub fn new(params: &KnpParams) -> Self {
        let p = params;
        assert!(p.tortuosity > 0.0 && p.volume_fraction > 0.0 && p.dt > 0.0);
        let h = [0, 1, 2].map(|a| p.extent[a] / p.cells[a] as f64);
        let grid = Grid::new_3d(p.cells, h);
        let n = grid.len();
        let d: Vec<f64> = (p.ions.iter())
            .map(|i| i.species.d / (p.tortuosity * p.tortuosity))
            .collect();
        let bc = match p.boundary {
            OuterBoundary::Dirichlet => BoundaryConditions::dirichlet(),
            OuterBoundary::NoFlux => BoundaryConditions::neumann(),
        };
        let laplace = DiffusionOperator::uniform(grid, bc, 1.0);
        let mut ghost = vec![0.0; n];
        laplace.apply(&vec![1.0; n], &mut ghost);
        // Forward Euler is stable for Δt ≤ 1 / max row sum of D̃(−Δ) (Gershgorin).
        let mut diag = vec![0.0; n];
        laplace.diagonal(&mut diag);
        let d_max = d.iter().cloned().fold(0.0, f64::max);
        let rate = d_max * diag.iter().cloned().fold(0.0, f64::max);
        let ions = p.ions.clone();
        Self {
            grid,
            c: ions.iter().map(|i| vec![i.baseline; n]).collect(),
            src: vec![vec![0.0; n]; ions.len()],
            z: ions.iter().map(|i| i.species.z as f64).collect(),
            ions,
            d,
            psi: psi_mv(p.temperature),
            alpha: p.volume_fraction,
            dt: p.dt,
            dt_max: if rate > 0.0 {
                1.0 / rate
            } else {
                f64::INFINITY
            },
            time: 0.0,
            c_next: vec![0.0; n],
            phi: vec![0.0; n],
            phi_prev: vec![0.0; n],
            cap: vec![0.0; n],
            charge_src: vec![0.0; n],
            source_cells: Vec::new(),
            is_source: vec![false; n],
            charge: vec![0.0; n],
            charge_ref: None,
            sigma: vec![0.0; n],
            zdc: vec![0.0; n],
            rhs: vec![0.0; n],
            drift: DriftFactors::new(n),
            mg: Multigrid::new(&laplace),
            mg_age: usize::MAX,
            op: laplace.clone(),
            laplace,
            ghost,
            pcg: Pcg::new(n),
            cg_opts: CgOptions {
                rtol: 0.0,
                atol: 0.0,
                max_iter: 200,
                null_space: match p.boundary {
                    OuterBoundary::Dirichlet => NullSpace::None,
                    OuterBoundary::NoFlux => NullSpace::Constant,
                },
            },
            stats: StepStats::default(),
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn ions(&self) -> &[Ion] {
        &self.ions
    }

    /// Concentration of ion `k` (mM).
    pub fn concentration(&self, k: usize) -> &[f64] {
        &self.c[k]
    }

    /// Mutable concentration of ion `k`, e.g. to set initial conditions. The charge
    /// density after the change becomes the new reference that steps preserve.
    pub fn concentration_mut(&mut self, k: usize) -> &mut [f64] {
        self.charge_ref = None;
        self.mg_age = usize::MAX;
        &mut self.c[k]
    }

    /// Potential (mV) from the last substep. It was solved for the concentrations at the
    /// start of that substep, so it lags [`concentration`](Self::concentration) by one.
    pub fn potential(&self) -> &[f64] {
        &self.phi
    }

    pub fn stats(&self) -> StepStats {
        self.stats
    }

    /// Largest stable explicit substep (ms).
    pub fn max_stable_dt(&self) -> f64 {
        self.dt_max
    }

    /// Total amount of ion `k` in the ECS, in amol (mM·µm³).
    pub fn total_amount(&self, k: usize) -> f64 {
        let v: f64 = self.grid.h.iter().product();
        self.alpha * v * self.c[k].iter().sum::<f64>()
    }

    /// Index of the cell containing `pos` (µm), or `None` outside the closed box.
    pub fn cell_at(&self, pos: [f64; 3]) -> Option<usize> {
        let g = &self.grid;
        let mut idx = [0; 3];
        for a in 0..3 {
            let t = pos[a] / g.h[a];
            if !(0.0..=g.n[a] as f64).contains(&t) {
                return None;
            }
            idx[a] = (t as usize).min(g.n[a] - 1);
        }
        Some(g.index(idx[0], idx[1], idx[2]))
    }

    /// Removes all point sources.
    pub fn clear_sources(&mut self) {
        for cell in self.source_cells.drain(..) {
            self.src.iter_mut().for_each(|s| s[cell] = 0.0);
            self.cap[cell] = 0.0;
            self.charge_src[cell] = 0.0;
            self.is_source[cell] = false;
        }
    }

    /// The cell containing `pos`, registered as a source cell.
    fn source_cell(&mut self, pos: [f64; 3]) -> Option<usize> {
        let cell = self.cell_at(pos)?;
        if !self.is_source[cell] {
            self.is_source[cell] = true;
            self.source_cells.push(cell);
        }
        Some(cell)
    }

    /// Adds a source of ion `k` releasing `rate` amol/ms (mM·µm³/ms) into the ECS at `pos`,
    /// binned into the containing cell. Sources persist until [`clear_sources`](Self::clear_sources).
    /// Returns false (and adds nothing) if `pos` is outside the box.
    ///
    /// With [`OuterBoundary::NoFlux`] the sources must carry no net current in total (as
    /// for a whole neuron); a net charge source has nowhere to go and is projected out of
    /// the potential equation, so the box slowly drifts from electroneutrality.
    pub fn add_ion_source(&mut self, pos: [f64; 3], k: usize, rate: f64) -> bool {
        let Some(cell) = self.source_cell(pos) else {
            return false;
        };
        let rate = rate / self.ecs_volume();
        self.src[k][cell] += rate;
        self.charge_src[cell] += self.z[k] * rate;
        true
    }

    /// Adds the membrane current (nA, positive out of the neuron) carried by ion `k`.
    /// Returns false for a neutral species, which carries no current.
    pub fn add_ion_current(&mut self, pos: [f64; 3], k: usize, current: f64) -> bool {
        self.z[k] != 0.0 && self.add_ion_source(pos, k, current * NA_TO_MM_UM3_PER_MS / self.z[k])
    }

    /// Adds a capacitive membrane current (nA, positive out of the neuron) at `pos`.
    pub fn add_capacitive_current(&mut self, pos: [f64; 3], current: f64) -> bool {
        let Some(cell) = self.source_cell(pos) else {
            return false;
        };
        let rate = current * NA_TO_MM_UM3_PER_MS / self.ecs_volume();
        self.cap[cell] += rate;
        self.charge_src[cell] += rate;
        true
    }

    /// ECS volume of one cell (µm³).
    fn ecs_volume(&self) -> f64 {
        self.alpha * self.grid.h.iter().product::<f64>()
    }

    /// Advances by one time step `dt`, split into equal substeps below the stability limit.
    pub fn step(&mut self) {
        let substeps = (self.dt / self.dt_max).ceil().max(1.0) as usize;
        let h = self.dt / substeps as f64;
        for _ in 0..substeps {
            self.substep(h);
        }
        self.stats.substeps = substeps;
        self.time += self.dt;
    }

    fn substep(&mut self, dt: f64) {
        // σ, Σ z c and Σ z D̃ (c − c_baseline) in one pass per species.
        self.sigma.fill(0.0);
        self.charge.fill(0.0);
        self.zdc.fill(0.0);
        for (k, ion) in self.ions.iter().enumerate() {
            let (d, z, base) = (self.d[k], self.z[k], ion.baseline);
            let w = d * z * z / self.psi;
            for (((s, q), t), c) in (self.sigma.iter_mut())
                .zip(&mut self.charge)
                .zip(&mut self.zdc)
                .zip(&self.c[k])
            {
                *s += w * c;
                *q += z * c;
                *t += z * d * (c - base);
            }
        }

        // A φ = Σ z f̃ + ĩ_cap/F − Σ z ∇·J_diff + (Σ z c − ρ_ref)/Δt, with A = −∇·(σ∇·)
        // and Σ z ∇·J_diff = −Δ(Σ z D̃ (c − c_baseline)).
        let charge_ref = (self.charge_ref).get_or_insert_with(|| self.charge.clone());
        self.laplace.apply(&self.zdc, &mut self.rhs);
        for (((r, s), q), q0) in (self.rhs.iter_mut())
            .zip(&self.charge_src)
            .zip(&self.charge)
            .zip(charge_ref.iter())
        {
            *r = s - *r + (q - q0) / dt;
        }
        self.stats = StepStats {
            converged: true,
            ..StepStats::default()
        };
        if self.sigma.iter().any(|&s| s > 0.0) {
            if self.mg_age >= OPERATOR_REFRESH_INTERVAL {
                self.op.update_sigma(&self.sigma);
                self.mg.update(&self.op);
                self.mg_age = 0;
            }
            self.mg_age += 1;
            for (p, q) in self.phi.iter_mut().zip(&mut self.phi_prev) {
                (*p, *q) = (2.0 * *p - *q, *p);
            }
            // Stop once the charge error this solve leaves behind, Δt ‖r‖, is negligible.
            self.cg_opts.atol = CHARGE_TOL / dt;
            let st = (self.pcg).solve(
                &self.op,
                &mut self.mg,
                &self.rhs,
                &mut self.phi,
                &self.cg_opts,
            );
            self.stats.converged = st.converged;
            self.stats.cg_iterations = st.iterations;
            self.stats.cg_residual = st.residual;
        }
        for (q0, cap) in charge_ref.iter_mut().zip(&self.cap) {
            *q0 -= dt * cap;
        }

        self.drift.update(
            &self.grid,
            self.psi,
            &self.sigma,
            self.op.sigma(),
            &self.phi,
            &self.ghost,
        );
        for k in 0..self.ions.len() {
            let fx = FaceFluxes {
                d: self.d[k],
                dz: self.d[k] * self.z[k],
                base: self.ions[k].baseline,
                inv_h2: self.grid.h.map(|h| 1.0 / (h * h)),
            };
            fx.advance(
                &self.grid,
                &self.drift,
                &self.ghost,
                &self.c[k],
                &self.src[k],
                dt,
                &mut self.c_next,
            );
            std::mem::swap(&mut self.c[k], &mut self.c_next);
        }
    }
}

impl Simulation for KnpSolver {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            KnpSolver::step(self);
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}
