//! A slab of tissue with thousands of reduced neurons coupled both ways to the KNP solver,
//! to ask whether extracellular ions alone can carry a spreading-depression wave.
//!
//! Each [`SdTissue::step`]: every compartment of the [`Population`] reads the Na⁺, K⁺ and X⁻
//! concentrations of the grid cell it sits in, the population advances over the KNP step,
//! and its mean currents become the KNP point sources. There are no synapses between the
//! cells: the only interaction is through the extracellular concentrations (the membranes do
//! not see the extracellular potential). Optional glial buffering takes up K⁺ together with
//! X⁻ (as KCl, so it carries no current) at a rate proportional to the excess over baseline;
//! it never releases K⁺ where [K⁺]e is below baseline. Optional astrocytes ([`crate::glia`]),
//! with or without that uptake, take up and redistribute K⁺ through their own membrane and
//! syncytium.
//!
//! Some neurons near the centre can be driven by an extra Na⁺ conductance on their soma
//! ([`SdTissue::set_drive`]). The question is whether that local K⁺ release can push the
//! neighbours over threshold so that the depolarisation spreads on its own.

use crate::glia::{Glia, GliaConfig};
use crate::population::{Population, PopulationParams};
use crate::tissue::{K, NA, X};
use crate::{KnpConfig, KnpSolver};
use ed_core::linalg::Boundary;
use ed_core::Simulation;

/// A dendrite above this potential (mV) counts as depolarised. Spikes barely reach the
/// dendrite, so this picks out sustained depolarisation.
pub const DEPOLARISED_MV: f64 = -40.0;

#[derive(Clone, Debug)]
pub struct SdConfig {
    /// Grid, extent and time step of the extracellular space. Boundaries are closed.
    pub knp: KnpConfig,
    pub neurons: usize,
    pub population: PopulationParams,
    /// Dendrite centre relative to its soma, µm.
    pub dendrite_offset: [f64; 3],
    /// Glial KCl uptake rate, 1/ms (0 = none).
    pub buffering: f64,
    /// An astrocytic syncytium in the tissue (spatial buffering), or none.
    pub glia: Option<GliaConfig>,
    /// Seed for placing the neurons.
    pub seed: u64,
}

impl Default for SdConfig {
    /// 10,000 neurons in a 1 × 1 × 0.2 mm slab (5·10⁴ per mm³, about half of cortex), somata
    /// in the lower half and dendrites 100 µm above them, on 25 µm cells with 4 ms steps.
    fn default() -> Self {
        Self {
            knp: KnpConfig {
                n: [40, 40, 8],
                size: [1000.0, 1000.0, 200.0],
                dt: 4.0,
                boundary: Boundary::Neumann,
                ..KnpConfig::default()
            },
            neurons: 10_000,
            population: PopulationParams {
                h_max: 4.0,
                ..PopulationParams::default()
            },
            dendrite_offset: [0.0, 0.0, 100.0],
            buffering: 0.0,
            glia: None,
            seed: 1,
        }
    }
}

pub struct SdTissue {
    pub knp: KnpSolver,
    pub pop: Population,
    /// Soma and dendrite positions per neuron (compartment `2i` and `2i + 1`), µm.
    positions: Vec<[f64; 3]>,
    /// Grid cell of each compartment.
    cells: Vec<usize>,
    buffering: f64,
    /// The astrocytic syncytium, if configured.
    pub glia: Option<Glia>,
    baseline: [f64; 3],
    /// α × cell volume, µm³: converts a concentration rate into an amount rate.
    cell_volume: f64,
    driven: Vec<usize>,
}

impl SdTissue {
    pub fn new(cfg: &SdConfig) -> Self {
        assert!(cfg.buffering >= 0.0, "buffering must be ≥ 0");
        let names: Vec<_> = cfg.knp.species.iter().map(|s| s.name).collect();
        assert!(
            names.len() > X && names[NA] == "Na+" && names[K] == "K+" && cfg.knp.species[X].z == -1,
            "the tissue must carry Na⁺, K⁺ and an anion first, got {names:?}"
        );
        assert_eq!(
            cfg.knp.temperature, cfg.population.temperature,
            "tissue and neurons must share one temperature"
        );
        let knp = KnpSolver::new(&cfg.knp);
        let size = cfg.knp.size;
        let off = cfg.dendrite_offset;
        let mut rng = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut uniform = move || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut positions = Vec::with_capacity(2 * cfg.neurons);
        for _ in 0..cfg.neurons {
            // Somata where both they and their dendrite fit in the box.
            let soma: [f64; 3] = std::array::from_fn(|a| {
                let lo = (-off[a]).max(0.0);
                let hi = size[a] - off[a].max(0.0);
                assert!(hi > lo, "dendrite offset does not fit in the box");
                lo + (hi - lo) * uniform()
            });
            positions.push(soma);
            // Rounding must not put the dendrite exactly on the far wall.
            positions.push([0, 1, 2].map(|a| (soma[a] + off[a]).min(size[a] * (1.0 - 1e-12))));
        }
        let cells = positions
            .iter()
            .map(|&p| knp.cell_at(p).expect("compartment inside the box"))
            .collect();
        let h = knp.grid().h;
        let b = &cfg.knp.baseline;
        let mut population = cfg.population.clone();
        population.ecs = [b[NA], b[K], b[X]];
        Self {
            pop: Population::new(cfg.neurons, population),
            positions,
            cells,
            buffering: cfg.buffering,
            glia: cfg.glia.as_ref().map(|g| Glia::new(g, &cfg.knp)),
            baseline: [b[NA], b[K], b[X]],
            cell_volume: cfg.knp.porosity * h[0] * h[1] * h[2],
            driven: Vec::new(),
            knp,
        }
    }

    /// Soma and dendrite positions (compartments `2i`, `2i + 1`), µm.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Drives the `count` neurons whose somata are closest to `centre` with `g_ns` nS each,
    /// and stops driving any others. Returns the driven neurons.
    pub fn set_drive(&mut self, centre: [f64; 3], count: usize, g_ns: f64) -> &[usize] {
        for &i in &self.driven {
            self.pop.set_drive(i, 0.0);
        }
        let dist = |i: usize| {
            let p = self.positions[2 * i];
            (0..3).map(|a| (p[a] - centre[a]).powi(2)).sum::<f64>()
        };
        let mut order: Vec<usize> = (0..self.pop.len()).collect();
        order.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
        order.truncate(count);
        for &i in &order {
            self.pop.set_drive(i, g_ns);
        }
        self.driven = order;
        &self.driven
    }

    /// The driven neurons.
    pub fn driven(&self) -> &[usize] {
        &self.driven
    }

    pub fn set_buffering(&mut self, rate: f64) {
        assert!(rate >= 0.0 && rate.is_finite(), "buffering must be ≥ 0");
        self.buffering = rate;
    }

    /// Fraction of neurons whose dendrite is above [`DEPOLARISED_MV`].
    pub fn depolarised_fraction(&self) -> f64 {
        let v = self.pop.v();
        let n = (1..v.len())
            .step_by(2)
            .filter(|&c| v[c] > DEPOLARISED_MV)
            .count();
        n as f64 / self.pop.len().max(1) as f64
    }
}

impl Simulation for SdTissue {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            for (c, &cell) in self.cells.iter().enumerate() {
                self.pop.na_e[c] = self.knp.concentration(NA)[cell];
                self.pop.k_e[c] = self.knp.concentration(K)[cell];
                self.pop.x_e[c] = self.knp.concentration(X)[cell];
            }
            self.pop.step(self.knp.dt());
            let s = &mut self.knp;
            s.clear_sources();
            let p = &self.pop;
            for (c, &cell) in self.cells.iter().enumerate() {
                s.add_ion_current_in_cell(cell, NA, p.i_na()[c]);
                s.add_ion_current_in_cell(cell, K, p.i_k()[c]);
                s.add_ion_current_in_cell(cell, X, p.i_x()[c]);
                s.add_capacitive_current_in_cell(cell, p.i_cap()[c]);
            }
            if self.buffering > 0.0 {
                // The exact relaxation over one step, so that the explicit source cannot
                // overshoot the baseline however large the rate.
                let dt = s.dt();
                let rate = (1.0 - (-self.buffering * dt).exp()) / dt * self.cell_volume;
                let k0 = self.baseline[1];
                for cell in 0..s.grid().len() {
                    let uptake = -rate * (s.concentration(K)[cell] - k0).max(0.0);
                    s.add_ion_source_in_cell(cell, K, uptake);
                    s.add_ion_source_in_cell(cell, X, uptake);
                }
            }
            if let Some(g) = self.glia.as_mut() {
                g.step(s);
            }
            s.step(1);
        }
    }

    fn time_ms(&self) -> f64 {
        self.knp.time_ms()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 400 neurons in a 200 × 200 × 200 µm box of 25 µm cells.
    fn small(g_nap: f64) -> SdConfig {
        let mut cfg = SdConfig {
            knp: KnpConfig {
                n: [8, 8, 8],
                size: [200.0, 200.0, 200.0],
                ..SdConfig::default().knp
            },
            neurons: 400,
            ..SdConfig::default()
        };
        cfg.population.g_nap = g_nap;
        cfg
    }

    /// Total amount of one species in the extracellular space, the neurons and the astrocytes,
    /// mM·µm³ (X⁻ outside pairs with Cl⁻ inside).
    fn amount(t: &SdTissue, cfg: &SdConfig, ion: usize) -> f64 {
        let h = t.knp.grid().h;
        let ecs: f64 =
            t.knp.concentration(ion).iter().sum::<f64>() * h[0] * h[1] * h[2] * cfg.knp.porosity;
        let glia = t
            .glia
            .as_ref()
            .zip(cfg.glia.as_ref())
            .map_or(0.0, |(g, c)| {
                let s = g.interior().concentration(ion);
                s.iter().sum::<f64>() * h[0] * h[1] * h[2] * c.volume_fraction
            });
        let p = &cfg.population;
        let inside = match ion {
            NA => t.pop.na_i(),
            K => t.pop.k_i(),
            _ => t.pop.cl_i(),
        };
        let ics: f64 = inside
            .iter()
            .enumerate()
            .map(|(c, x)| x * p.volume[c % 2])
            .sum();
        ecs + ics + glia
    }

    #[test]
    fn undriven_tissue_stays_at_rest() {
        let cfg = small(0.25);
        let mut t = SdTissue::new(&cfg);
        t.step(250);
        let k0 = cfg.knp.baseline[K];
        assert!(t
            .knp
            .concentration(K)
            .iter()
            .all(|&c| (c - k0).abs() < 1e-6));
        assert_eq!(t.depolarised_fraction(), 0.0);
        assert!(t
            .positions()
            .iter()
            .all(|p| (0..3).all(|a| (0.0..200.0).contains(&p[a]))));
    }

    #[test]
    fn ions_are_conserved_between_cells_and_tissue() {
        let cfg = small(0.25);
        let mut t = SdTissue::new(&cfg);
        assert_eq!(t.set_drive([100.0, 100.0, 50.0], 2, 20.0).len(), 2);
        let before = [NA, K, X].map(|ion| amount(&t, &cfg, ion));
        t.step(100);
        assert!(
            t.knp.concentration(K).iter().any(|&c| c > 3.5),
            "the driven cells release K⁺"
        );
        for (ion, a0) in [NA, K, X].into_iter().zip(before) {
            let a1 = amount(&t, &cfg, ion);
            assert!((a1 - a0).abs() < 1e-9 * a0, "species {ion}: {a0} → {a1}");
        }
        // Buffering removes K⁺ and X⁻ in equal amounts.
        let (k1, x1) = (amount(&t, &cfg, K), amount(&t, &cfg, X));
        t.set_buffering(1e-3);
        t.step(100);
        let (dk, dx) = (k1 - amount(&t, &cfg, K), x1 - amount(&t, &cfg, X));
        assert!(dk > 0.0 && (dk - dx).abs() < 1e-9 * k1, "{dk} vs {dx}");
    }

    #[test]
    fn astrocytes_conserve_ions_and_take_up_the_released_potassium() {
        let run = |glia: Option<GliaConfig>| {
            let cfg = SdConfig {
                glia,
                ..small(0.25)
            };
            let mut t = SdTissue::new(&cfg);
            t.set_drive([100.0, 100.0, 50.0], 2, 20.0);
            let before = [NA, K, X].map(|ion| amount(&t, &cfg, ion));
            t.step(250);
            for (ion, a0) in [NA, K, X].into_iter().zip(before) {
                let a1 = amount(&t, &cfg, ion);
                assert!((a1 - a0).abs() < 1e-9 * a0, "species {ion}: {a0} → {a1}");
            }
            let k_max = t
                .knp
                .concentration(K)
                .iter()
                .fold(0.0, |m: f64, &c| m.max(c));
            (t, k_max)
        };
        let (_, bare) = run(None);
        let (t, glia) = run(Some(GliaConfig::default()));
        assert!(
            bare > 4.0 && glia < 3.0 + 0.5 * (bare - 3.0),
            "{bare} vs {glia}"
        );
        let g = t.glia.as_ref().unwrap();
        let k_i = g.interior().concentration(K);
        assert!(k_i.iter().any(|&k| k > g.membrane().params().ics[1] + 0.1));
    }

    #[test]
    fn strong_buffering_stays_stable() {
        let cfg = small(0.25);
        let mut t = SdTissue::new(&cfg);
        t.set_drive([100.0, 100.0, 50.0], 2, 20.0);
        t.set_buffering(1.0); // 4 per step: an explicit uptake would diverge.
        t.step(200);
        let k0 = cfg.knp.baseline[K];
        let k = t.knp.concentration(K);
        let (lo, hi) = k
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &c| (l.min(c), h.max(c)));
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo > k0 - 0.5 && hi < k0 + 0.5, "[K⁺]e within {lo}–{hi}");
    }

    #[test]
    fn drive_moves_to_the_nearest_somata() {
        let mut t = SdTissue::new(&small(0.0));
        let near = |t: &SdTissue, centre: [f64; 3]| {
            let d = |i: usize| {
                let p = t.positions()[2 * i];
                (0..3).map(|a| (p[a] - centre[a]).powi(2)).sum::<f64>()
            };
            let far = t.driven().iter().map(|&i| d(i)).fold(0.0, f64::max);
            (0..t.pop.len()).filter(|&i| d(i) <= far).count()
        };
        t.set_drive([20.0, 20.0, 20.0], 5, 20.0);
        assert_eq!(near(&t, [20.0, 20.0, 20.0]), 5);
        let old = t.driven().to_vec();
        t.set_drive([180.0, 180.0, 50.0], 3, 20.0);
        assert_eq!(near(&t, [180.0, 180.0, 50.0]), 3);
        assert!(old.iter().all(|i| !t.driven().contains(i)));
        // The old ones are released: their somata recover while the new ones depolarise.
        t.step(100);
        let v = t.pop.v();
        assert!(old.iter().all(|&i| v[2 * i] < -55.0));
    }

    #[test]
    fn one_driven_cell_ignites_an_excitable_tissue_but_not_a_robust_one() {
        let run = |g_nap: f64| {
            let mut t = SdTissue::new(&small(g_nap));
            t.set_drive([100.0, 100.0, 50.0], 1, 20.0);
            t.step(1000); // 4 s
            t.depolarised_fraction()
        };
        assert!(run(0.3) > 0.9);
        assert!(run(0.0) < 0.01);
    }
}
