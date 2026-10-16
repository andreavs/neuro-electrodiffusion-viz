//! A slab of tissue with thousands of reduced neurons coupled both ways to the KNP solver,
//! to ask whether extracellular ions alone can carry a spreading-depression wave.
//!
//! Each [`SdTissue::step`]: every compartment of the [`Population`] reads the Na⁺, K⁺ and X⁻
//! concentrations of the grid cell it sits in, the population advances over the KNP step,
//! and its mean currents become the KNP point sources. There are no synapses between the
//! cells: the only interaction is through the extracellular concentrations (and the small
//! extracellular potential). Optional glial buffering takes up K⁺ together with X⁻ (as KCl,
//! so it carries no current) at a rate proportional to the excess over baseline.
//!
//! Some neurons near the centre can be driven by an extra Na⁺ conductance on their soma
//! ([`SdTissue::set_drive`]). The question is whether that local K⁺ release can push the
//! neighbours over threshold so that the depolarisation spreads on its own.

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
    baseline: [f64; 3],
    /// α × cell volume, µm³: converts a concentration rate into an amount rate.
    cell_volume: f64,
    driven: Vec<usize>,
}

impl SdTissue {
    pub fn new(cfg: &SdConfig) -> Self {
        assert!(cfg.buffering >= 0.0, "buffering must be ≥ 0");
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
            positions.push([0, 1, 2].map(|a| soma[a] + off[a]));
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
                let (k0, rate) = (self.baseline[1], self.buffering * self.cell_volume);
                for cell in 0..s.grid().len() {
                    let uptake = -rate * (s.concentration(K)[cell] - k0);
                    s.add_ion_source_in_cell(cell, K, uptake);
                    s.add_ion_source_in_cell(cell, X, uptake);
                }
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

    /// Total amount of K⁺ in the extracellular space and the cells, mM·µm³.
    fn potassium(t: &SdTissue, cfg: &SdConfig) -> f64 {
        let h = t.knp.grid().h;
        let ecs: f64 =
            t.knp.concentration(K).iter().sum::<f64>() * h[0] * h[1] * h[2] * cfg.knp.porosity;
        let p = &cfg.population;
        let ics: f64 = t
            .pop
            .k_i()
            .iter()
            .enumerate()
            .map(|(c, k)| k * p.volume[c % 2])
            .sum();
        ecs + ics
    }

    #[test]
    fn undriven_tissue_stays_at_rest() {
        let cfg = small(0.25);
        let mut t = SdTissue::new(&cfg);
        t.step(250);
        let k = t.knp.concentration(K);
        assert!(k.iter().all(|&c| (c - 3.0).abs() < 1e-6));
        assert_eq!(t.depolarised_fraction(), 0.0);
        assert!(t
            .positions()
            .iter()
            .all(|p| (0..3).all(|a| (0.0..200.0).contains(&p[a]))));
    }

    #[test]
    fn potassium_is_conserved_between_cells_and_tissue() {
        let cfg = small(0.25);
        let mut t = SdTissue::new(&cfg);
        assert_eq!(t.set_drive([100.0, 100.0, 50.0], 2, 20.0).len(), 2);
        let k0 = potassium(&t, &cfg);
        t.step(100);
        assert!(
            t.knp.concentration(K).iter().any(|&c| c > 3.5),
            "the driven cells release K⁺"
        );
        let k1 = potassium(&t, &cfg);
        assert!((k1 - k0).abs() < 1e-9 * k0, "{k0} → {k1}");
        // Buffering removes K⁺ from the system.
        t.set_buffering(1e-3);
        t.step(100);
        assert!(potassium(&t, &cfg) < k1);
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
