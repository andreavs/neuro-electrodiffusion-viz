//! An astrocytic syncytium on the KNP grid: the astrocytes' interior as a second
//! electrodiffusive domain, coupled to the extracellular space through the membrane of
//! [`crate::astrocyte`], after the model of Halnes et al. (2013).
//!
//! Every grid cell holds both domains: the extracellular space of the [`KnpSolver`] the
//! astrocytes are attached to, and the astrocytic interior with volume fraction α_g, joined by gap
//! junctions into one syncytium. The interior is a [`KnpSolver`] of its own on the same grid,
//! with Na⁺, K⁺ and Cl⁻ (at the indices [`NA`], [`K`] and [`X`]) and tortuosity λ_g. Each step:
//!
//! 1. The membrane potential V solves the cable equation, backward Euler with the chord
//!    conductances (at the old V), reversal potentials and pump frozen over the step:
//!    `a C_m ∂V/∂t + a [Σ_k g_k (V − E_k) + I_p] = ∇·(σ_m ∇V)`, `σ_m = σ_g σ_E / (σ_g + σ_E)`,
//!    where `a` is the membrane area per tissue volume and σ_g, σ_E are the conductivities of
//!    the two domains per tissue volume. This is the bidomain model reduced to one equation,
//!    exact when σ_g/σ_E is uniform; it leaves out diffusion potentials and the currents of
//!    other cells in the extracellular space. V is held at rest on Dirichlet boundaries.
//! 2. The ionic and capacitive membrane currents at the new V become sources of both domains,
//!    with opposite signs, so every ion that leaves the astrocytes enters the extracellular
//!    space of the same grid cell and both domains stay electroneutral.
//! 3. The interior takes a KNP step. The caller then steps the extracellular solver.
//!
//! The membrane currents use the concentrations at the start of the step, so a step must not
//! move a sizeable part of the extracellular K⁺: with the defaults, steps above about 60 ms
//! make [K⁺]e overshoot.
//!
//! This gives spatial buffering: where [K⁺]e is high, E_K rises above V and K⁺ flows in; the
//! depolarisation spreads along the syncytium, and where [K⁺]e is at baseline the depolarised
//! membrane releases K⁺.

use crate::astrocyte::{AstrocyteMembrane, AstrocyteParams};
use crate::tissue::{K, NA, X};
use crate::{KnpConfig, KnpSolver};
use ed_core::constants::F;
use ed_core::ions;
use ed_core::linalg::{CgOptions, DiffusionOperator, Multigrid, NullSpace, Pcg};
use ed_core::Simulation;

/// Parameters of the astrocytic syncytium.
#[derive(Clone, Debug)]
pub struct GliaConfig {
    /// The membrane. Its extracellular baseline and temperature are taken from the
    /// extracellular space.
    pub membrane: AstrocyteParams,
    /// Astrocytic volume fraction α_g.
    pub volume_fraction: f64,
    /// Tortuosity of the astrocytic interior.
    pub tortuosity: f64,
    /// Astrocytic membrane area per tissue volume, µm⁻¹.
    pub area_per_volume: f64,
    /// Scales diffusion and conduction through the interior: 1 is a fully coupled syncytium,
    /// smaller values mimic blocked gap junctions. Must be positive.
    pub coupling: f64,
}

impl Default for GliaConfig {
    /// Halnes et al. (2013): α_g = 0.4, λ_g = 3.2 and 8 µm² of membrane per µm³ of tissue
    /// (0.05 µm of astrocyte volume per membrane area).
    fn default() -> Self {
        Self {
            membrane: AstrocyteParams::default(),
            volume_fraction: 0.4,
            tortuosity: 3.2,
            area_per_volume: 8.0,
            coupling: 1.0,
        }
    }
}

/// Astrocytes filling the grid of an extracellular [`KnpSolver`]. See the [module docs](self).
pub struct Glia {
    interior: KnpSolver,
    membrane: AstrocyteMembrane,
    /// Membrane potential per cell, mV.
    v: Vec<f64>,
    /// Chord conductances (mS/cm²) and reversal potentials (mV) of Na⁺, K⁺ and Cl⁻, and the
    /// pump current (µA/cm²), per cell, frozen over a step.
    g: Vec<[f64; 3]>,
    e: Vec<[f64; 3]>,
    pump: Vec<f64>,
    /// Converts a membrane current density (µA/cm²) into a charge rate per tissue volume
    /// (mM/ms, the units of the KNP potential equation) and into a current per cell (nA).
    to_rate: f64,
    to_na: f64,
    /// The cable operator `shift − ∇·σ_m∇` acting on V − V_rest.
    op: DiffusionOperator,
    mg: Multigrid,
    pcg: Pcg,
    opts: CgOptions,
    sigma: Vec<f64>,
    sigma_e: Vec<f64>,
    shift: Vec<f64>,
    rhs: Vec<f64>,
    u: Vec<f64>,
}

impl Glia {
    /// Astrocytes at rest in the extracellular space described by `knp`, which must carry
    /// Na⁺, K⁺ and an anion first (as [`crate::tissue`] does).
    pub fn new(cfg: &GliaConfig, knp: &KnpConfig) -> Self {
        let names: Vec<_> = knp.species.iter().map(|s| s.name).collect();
        assert!(
            names.len() > X && names[NA] == "Na+" && names[K] == "K+" && knp.species[X].z == -1,
            "the extracellular space must carry Na⁺, K⁺ and an anion first, got {names:?}"
        );
        assert!(
            cfg.volume_fraction > 0.0
                && cfg.tortuosity > 0.0
                && cfg.area_per_volume > 0.0
                && cfg.coupling > 0.0,
            "volume fraction, tortuosity, membrane area and coupling must be positive"
        );
        let b = &knp.baseline;
        let membrane = AstrocyteMembrane::new(AstrocyteParams {
            ecs: [b[NA], b[K], b[X]],
            temperature: knp.temperature,
            ..cfg.membrane.clone()
        });
        let cal = membrane.calibration();
        let [na_i, k_i] = membrane.params().ics;
        let interior = KnpSolver::new(&KnpConfig {
            species: vec![ions::NA, ions::K, ions::CL],
            baseline: vec![na_i, k_i, cal.cl_i],
            tortuosity: cfg.tortuosity / cfg.coupling.sqrt(),
            porosity: cfg.volume_fraction,
            ..knp.clone()
        });
        let (grid, len) = (interior.model.grid, interior.model.grid.len());
        let op = DiffusionOperator::uniform(grid, interior.model.bc, 1.0);
        let h = grid.h;
        Self {
            v: vec![cal.v_rest; len],
            g: vec![[0.0; 3]; len],
            e: vec![[0.0; 3]; len],
            pump: vec![0.0; len],
            // 1 µA/cm² = 1e-17 C/(ms·µm²) and 1 mM = 1e-18 mol/µm³; 1 µA/cm² = 1e-5 nA/µm².
            to_rate: 10.0 * cfg.area_per_volume / F,
            to_na: 1e-5 * cfg.area_per_volume * h[0] * h[1] * h[2],
            mg: Multigrid::new(&op),
            op,
            pcg: Pcg::new(len),
            opts: CgOptions {
                rtol: 1e-10,
                atol: 1e-15 * (len as f64).sqrt(),
                max_iter: 500,
                null_space: NullSpace::None,
            },
            sigma: vec![0.0; len],
            sigma_e: vec![0.0; len],
            shift: vec![0.0; len],
            rhs: vec![0.0; len],
            u: vec![0.0; len],
            interior,
            membrane,
        }
    }

    /// The astrocytic interior, with Na⁺, K⁺ and Cl⁻ at [`NA`], [`K`] and [`X`].
    pub fn interior(&self) -> &KnpSolver {
        &self.interior
    }

    /// The membrane, calibrated to the extracellular baseline.
    pub fn membrane(&self) -> &AstrocyteMembrane {
        &self.membrane
    }

    /// Membrane potential per grid cell, mV.
    pub fn potential(&self) -> &[f64] {
        &self.v
    }

    /// Advances the astrocytes by one step of `ecs`: solves for the membrane potential, adds
    /// the membrane currents to the sources of `ecs` and steps the interior. Clear the sources
    /// of `ecs` before calling, and step `ecs` afterwards.
    pub fn step(&mut self, ecs: &mut KnpSolver) {
        let len = self.v.len();
        assert!(ecs.grid() == self.interior.grid(), "grids must match");
        let dt = ecs.dt();
        if self.interior.dt() != dt {
            self.interior.set_dt(dt);
        }
        let m = &self.membrane;
        let (v_rest, c_m) = (m.calibration().v_rest, m.params().c_m);
        let (ce, ci) = (&ecs.c, &self.interior.c);
        for p in 0..len {
            let ecs_p = [ce[NA][p], ce[K][p], ce[X][p]];
            let e = m.reversal(ecs_p, [ci[NA][p], ci[K][p], ci[X][p]]);
            let g = m.conductances(self.v[p], e, ecs_p[1]);
            let pump = m.pump(ecs_p[1], ci[NA][p]);
            let gs = g[0] + g[1] + g[2];
            let ge: f64 = (0..3).map(|k| g[k] * (e[k] - v_rest)).sum();
            self.shift[p] = self.to_rate * (c_m / dt + gs);
            self.rhs[p] = self.to_rate * (c_m / dt * (self.v[p] - v_rest) + ge - pump);
            self.u[p] = self.v[p] - v_rest;
            (self.g[p], self.e[p], self.pump[p]) = (g, e, pump);
        }
        self.interior.conductivity(&mut self.sigma);
        ecs.conductivity(&mut self.sigma_e);
        for (s, se) in self.sigma.iter_mut().zip(&self.sigma_e) {
            let sum = *s + se;
            *s = if sum > 0.0 { *s * se / sum } else { 0.0 };
        }
        self.op.update_sigma(&self.sigma);
        self.op.set_shift(&self.shift);
        self.mg.update(&self.op);
        let stats = self
            .pcg
            .solve(&self.op, &mut self.mg, &self.rhs, &mut self.u, &self.opts);
        debug_assert!(stats.converged, "cable solve: {stats:?}");

        self.interior.clear_sources();
        for p in 0..len {
            let v = v_rest + self.u[p];
            let (g, e, pump) = (self.g[p], self.e[p], self.pump[p]);
            let i = [
                g[0] * (v - e[0]) + 3.0 * pump,
                g[1] * (v - e[1]) - 2.0 * pump,
                g[2] * (v - e[2]),
            ];
            for (k, ik) in [NA, K, X].into_iter().zip(i) {
                ecs.add_ion_current_in_cell(p, k, ik * self.to_na);
                self.interior
                    .add_ion_current_in_cell(p, k, -ik * self.to_na);
            }
            let i_cap = c_m * (v - self.v[p]) / dt * self.to_na;
            ecs.add_capacitive_current_in_cell(p, i_cap);
            self.interior.add_capacitive_current_in_cell(p, -i_cap);
            self.v[p] = v;
        }
        self.interior.step(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed_core::linalg::Boundary;

    /// A closed 300 µm column of 10 µm cells, as Halnes et al.'s astrocyte.
    fn column(dt: f64) -> KnpConfig {
        KnpConfig {
            n: [30, 1, 1],
            size: [300.0, 10.0, 10.0],
            dt,
            boundary: Boundary::Neumann,
            ..KnpConfig::default()
        }
    }

    struct Tissue {
        ecs: KnpSolver,
        glia: Option<Glia>,
        cfg: KnpConfig,
    }

    impl Tissue {
        fn new(cfg: KnpConfig, glia: Option<GliaConfig>) -> Self {
            Self {
                ecs: KnpSolver::new(&cfg),
                glia: glia.map(|g| Glia::new(&g, &cfg)),
                cfg,
            }
        }

        /// Halnes et al.'s input: neurons in the first tenth release K⁺ for Na⁺ at 22 mM/s
        /// of extracellular volume, and everywhere take the K⁺ excess back at 1.16 /s.
        fn step(&mut self, seconds: f64, input: bool) {
            let h = self.ecs.grid().h;
            let vol = self.cfg.porosity * h[0] * h[1] * h[2];
            let (n, k0) = (self.ecs.grid().len(), self.cfg.baseline[K]);
            for _ in 0..(seconds * 1e3 / self.cfg.dt).round() as usize {
                self.ecs.clear_sources();
                if input {
                    for p in 0..n {
                        let rate = if p < n / 10 { 0.022 } else { 0.0 };
                        let r = (rate - 1.16e-3 * (self.ecs.concentration(K)[p] - k0)) * vol;
                        self.ecs.add_ion_source_in_cell(p, K, r);
                        self.ecs.add_ion_source_in_cell(p, NA, -r);
                    }
                }
                if let Some(g) = self.glia.as_mut() {
                    g.step(&mut self.ecs);
                }
                self.ecs.step(1);
            }
        }

        fn k_e(&self) -> &[f64] {
            self.ecs.concentration(K)
        }

        /// Amount of Na⁺, K⁺ and X⁻/Cl⁻ in both domains, mM·µm³.
        fn amounts(&self) -> [f64; 3] {
            let h = self.ecs.grid().h;
            let cell = h[0] * h[1] * h[2];
            let sum = |s: &KnpSolver, k: usize| s.concentration(k).iter().sum::<f64>() * cell;
            [NA, K, X].map(|k| {
                let glia = self
                    .glia
                    .as_ref()
                    .map_or(0.0, |g| g.interior.porosity * sum(g.interior(), k));
                self.cfg.porosity * sum(&self.ecs, k) + glia
            })
        }
    }

    #[test]
    fn rest_is_a_steady_state() {
        let cfg = KnpConfig {
            n: [6, 6, 4],
            size: [150.0, 150.0, 100.0],
            dt: 4.0,
            ..KnpConfig::default()
        };
        let mut t = Tissue::new(cfg.clone(), Some(GliaConfig::default()));
        t.step(0.2, false);
        let g = t.glia.as_ref().unwrap();
        let v_rest = g.membrane().calibration().v_rest;
        assert!(g.potential().iter().all(|v| (v - v_rest).abs() < 1e-9));
        let ics = g.interior();
        let cl_i = g.membrane().calibration().cl_i;
        let [na_i, k_i] = g.membrane().params().ics;
        for (s, base) in [(&t.ecs, &cfg.baseline[..3]), (ics, &[na_i, k_i, cl_i][..])] {
            for (k, b) in base.iter().enumerate() {
                assert!(s.concentration(k).iter().all(|c| (c - b).abs() < 1e-9));
            }
        }
    }

    #[test]
    fn ions_are_conserved_between_the_domains() {
        let mut t = Tissue::new(column(10.0), Some(GliaConfig::default()));
        // A KCl bump at one end.
        for k in [K, X] {
            t.ecs.concentration_mut(k)[..3]
                .iter_mut()
                .for_each(|c| *c += 5.0);
        }
        let before = t.amounts();
        t.step(5.0, false);
        let g = t.glia.as_ref().unwrap();
        assert!(g.interior().concentration(K)[0] > g.membrane().params().ics[1] + 0.5);
        for (a0, a1) in before.into_iter().zip(t.amounts()) {
            assert!((a1 - a0).abs() < 1e-10 * a0, "{a0} → {a1}");
        }
    }

    #[test]
    fn astrocytes_carry_potassium_away_from_the_input() {
        let run = |glia: Option<GliaConfig>| {
            let mut t = Tissue::new(column(10.0), glia);
            t.step(20.0, true);
            t
        };
        let bare = run(None);
        let glia = run(Some(GliaConfig::default()));
        let blocked = run(Some(GliaConfig {
            coupling: 0.01,
            ..GliaConfig::default()
        }));
        let (near, far, k0) = (0, 29, 3.0);
        // The syncytium takes up K⁺ at the input and raises it far away, where diffusion
        // through the extracellular space alone has not yet reached in 20 s.
        assert!(glia.k_e()[near] < 0.75 * bare.k_e()[near]);
        assert!(bare.k_e()[far] < k0 + 0.05 && glia.k_e()[far] > k0 + 0.15);
        // Blocking the gap junctions leaves the K⁺ where it was released.
        assert!(blocked.k_e()[near] > 0.9 * bare.k_e()[near]);
        assert!(blocked.k_e()[far] < k0 + 0.05);

        let g = glia.glia.as_ref().unwrap();
        let v = g.potential();
        assert!(
            v[near] > -70.0 && v.windows(2).all(|w| w[0] > w[1]),
            "{v:?}"
        );
        let i_k = |p: usize| {
            let (e, i) = (&glia.ecs, g.interior());
            let ecs = [NA, K, X].map(|k| e.concentration(k)[p]);
            let ics = [NA, K, X].map(|k| i.concentration(k)[p]);
            g.membrane().currents(v[p], ecs, ics)[1]
        };
        assert!(
            i_k(near) < 0.0 && i_k(far) > 0.0,
            "uptake near, release far"
        );

        // The cable potential agrees with the two domain potentials of the KNP solvers,
        // which include the diffusion potentials the cable leaves out.
        let dphi = |p: usize| g.interior().potential()[p] - glia.ecs.potential()[p];
        let (dv, dknp) = (v[near] - v[far], dphi(near) - dphi(far));
        assert!(
            (dv - dknp).abs() < 0.1 * dv,
            "ΔV {dv} vs Δ(φ_g − φ_E) {dknp}"
        );
    }

    #[test]
    fn membrane_charge_sits_on_the_capacitor() {
        // Dirichlet ends, which hold V at rest beyond the input.
        let cfg = KnpConfig {
            boundary: Boundary::Dirichlet,
            ..column(10.0)
        };
        let mut t = Tissue::new(cfg.clone(), Some(GliaConfig::default()));
        t.step(4.0, true);
        let g = t.glia.as_ref().unwrap();
        let (v, m) = (g.potential(), g.membrane());
        let v_rest = m.calibration().v_rest;
        assert!(v[0] < v[2] && v[2] > v_rest + 3.0, "{v:?}");
        // The charge each domain gained (mM per tissue volume) is the plate charge a C_m ΔV.
        let a = GliaConfig::default().area_per_volume;
        let charge = |s: &KnpSolver, z: &[f64], p: usize| {
            s.porosity
                * z.iter()
                    .enumerate()
                    .map(|(k, z)| z * s.concentration(k)[p])
                    .sum::<f64>()
        };
        let z_ecs: Vec<f64> = cfg.species.iter().map(|s| s.z as f64).collect();
        let [na_i, k_i] = m.params().ics;
        let rho_g0 = g.interior().porosity * (na_i + k_i - m.calibration().cl_i);
        let plate = |p: usize| 10.0 * a * m.params().c_m * (v[p] - v_rest) / F;
        let max = plate(2);
        for p in 0..30 {
            let ecs = charge(&t.ecs, &z_ecs, p);
            let glia = charge(g.interior(), &[1.0, 1.0, -1.0], p) - rho_g0;
            assert!(
                (ecs + plate(p)).abs() < 0.01 * max,
                "cell {p}: {ecs} vs {}",
                plate(p)
            );
            assert!(
                (glia - plate(p)).abs() < 0.01 * max,
                "cell {p}: {glia} vs {}",
                plate(p)
            );
        }
    }

    #[test]
    fn the_time_step_barely_matters() {
        let run = |dt: f64| {
            let mut t = Tissue::new(column(dt), Some(GliaConfig::default()));
            t.step(4.0, true);
            (t.k_e()[0], t.glia.unwrap().potential()[0])
        };
        let ((k1, v1), (k10, v10)) = (run(1.0), run(10.0));
        assert!((k1 - k10).abs() < 0.01 * (k1 - 3.0), "{k1} vs {k10}");
        assert!((v1 - v10).abs() < 0.2, "{v1} vs {v10}");
    }
}
