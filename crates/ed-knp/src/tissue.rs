//! A neuron coupled one way to the tissue: its membrane currents are the KNP point sources.
//!
//! Each [`Tissue::step`] advances the neuron over one KNP time step in internal steps of
//! its own Δt, averages its currents over the interval, deposits them at the compartment
//! centres and then takes one KNP step. Concentrations seen by the neuron stay fixed, as in
//! Publication 1. The electrode return current (see [`Neuron::electrode_return`]) is
//! released as X⁻ at the soma, as from a Ag/AgCl ground, so the net membrane current
//! still sums to zero.

use crate::neuron::Neuron;
use crate::{KnpConfig, KnpSolver};
use ed_core::linalg::Boundary;
use ed_core::Simulation;

/// Species indices in the default [`KnpConfig`].
pub const NA: usize = 0;
pub const K: usize = 1;
pub const X: usize = 2;
pub const CA: usize = 3;

/// Somatic potential (mV) an upward crossing of which counts as a spike.
const SPIKE_THRESHOLD: f64 = 0.0;
/// Trace samples kept when nobody drains them (the newest win), so an unread trace cannot
/// grow forever.
const MAX_TRACE: usize = 1 << 16;
/// Longest KNP step accepted, ms.
pub const MAX_DT: f64 = 50.0;

#[derive(Clone, Debug)]
pub struct TissueConfig {
    pub knp: KnpConfig,
    /// Where the soma (the neuron's origin) sits in the grid, µm.
    pub soma: [f64; 3],
    /// Neuron internal steps per somatic trace sample. Spikes are detected on these
    /// samples, so keep it at a fraction of a millisecond.
    pub trace_every: usize,
}

impl Default for TissueConfig {
    /// A 640 × 640 × 1280 µm column of 20 µm cells (32 × 32 × 64) around the default
    /// pyramidal cell, with baseline concentrations held on the outer boundary and 1 ms
    /// KNP steps.
    fn default() -> Self {
        Self {
            knp: KnpConfig {
                n: [32, 32, 64],
                size: [640.0, 640.0, 1280.0],
                dt: 1.0,
                boundary: Boundary::Dirichlet,
                ..KnpConfig::default()
            },
            soma: [320.0, 320.0, 190.0],
            trace_every: 4,
        }
    }
}

pub struct Tissue {
    pub knp: KnpSolver,
    pub neuron: Neuron,
    soma: [f64; 3],
    /// Compartment centres in grid coordinates (µm).
    positions: Vec<[f64; 3]>,
    trace_every: usize,
    /// Current sums over the step: Na⁺, K⁺, X⁻, capacitive, per compartment.
    acc: [Vec<f64>; 4],
    /// Somatic (t, V) samples not yet taken.
    trace: Vec<f64>,
    spikes: u64,
    above: bool,
}

impl Tissue {
    /// Couples `neuron` to a new KNP solver. Panics if a compartment lies outside the grid.
    pub fn new(cfg: &TissueConfig, neuron: Neuron) -> Self {
        assert!(cfg.trace_every > 0, "trace_every must be positive");
        let mut knp = KnpSolver::new(&cfg.knp);
        knp.set_dt(whole_steps(cfg.knp.dt, neuron.params().dt));
        let positions: Vec<[f64; 3]> = neuron
            .centres()
            .iter()
            .map(|c| [0, 1, 2].map(|a| c[a] + cfg.soma[a]))
            .collect();
        for p in &positions {
            assert!(
                knp.cell_at(*p).is_some(),
                "compartment at {p:?} is outside the grid"
            );
        }
        let n = neuron.len();
        let above = neuron.v()[0] > SPIKE_THRESHOLD;
        Self {
            knp,
            soma: cfg.soma,
            positions,
            trace_every: cfg.trace_every,
            acc: std::array::from_fn(|_| vec![0.0; n]),
            trace: Vec::new(),
            spikes: 0,
            above,
            neuron,
        }
    }

    /// The default column with the default pyramidal cell.
    pub fn pyramidal() -> Self {
        Self::new(&TissueConfig::default(), Neuron::pyramidal())
    }

    /// Soma position in grid coordinates, µm.
    pub fn soma(&self) -> [f64; 3] {
        self.soma
    }

    /// Compartment centres in grid coordinates, µm.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Spikes at the soma so far.
    pub fn spikes(&self) -> u64 {
        self.spikes
    }

    /// Hands over the somatic trace as (t ms, V mV) pairs recorded since the last call.
    pub fn take_trace(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.trace)
    }

    /// Sets the KNP time step (ms), rounded to a whole number of neuron steps. Panics
    /// unless `0 < dt <= MAX_DT`.
    pub fn set_dt(&mut self, dt: f64) {
        let dt = whole_steps(dt, self.neuron.params().dt);
        self.knp.set_dt(dt);
    }

    /// Neuron steps per KNP step.
    fn substeps(&self) -> usize {
        (self.knp.dt() / self.neuron.params().dt).round().max(1.0) as usize
    }

    fn record(&mut self) {
        let v = self.neuron.v()[0];
        let up = v > SPIKE_THRESHOLD;
        if up && !self.above {
            self.spikes += 1;
        }
        self.above = up;
        if self.trace.len() >= 2 * MAX_TRACE {
            self.trace.drain(..MAX_TRACE);
        }
        self.trace.extend([self.neuron.time_ms(), v]);
    }

    /// Advances the neuron over one KNP step and deposits its mean currents.
    fn advance_neuron(&mut self) -> f64 {
        let total = self.substeps();
        self.acc.iter_mut().for_each(|a| a.fill(0.0));
        let mut electrode = 0.0;
        let mut done = 0;
        while done < total {
            let m = self.trace_every.min(total - done);
            self.neuron.step(m);
            let w = m as f64 / total as f64;
            let n = &self.neuron;
            for (acc, cur) in self
                .acc
                .iter_mut()
                .zip([n.i_na(), n.i_k(), n.i_x(), n.i_cap()])
            {
                acc.iter_mut().zip(cur).for_each(|(a, &c)| *a += w * c);
            }
            electrode += w * n.electrode_return();
            done += m;
            self.record();
        }
        electrode
    }
}

/// `dt` rounded to a whole number (at least one) of neuron steps `h`.
fn whole_steps(dt: f64, h: f64) -> f64 {
    assert!(
        dt > 0.0 && dt <= MAX_DT,
        "KNP time step must be in (0, {MAX_DT}] ms"
    );
    (dt / h).round().max(1.0) * h
}

impl Simulation for Tissue {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            let electrode = self.advance_neuron();
            let s = &mut self.knp;
            s.clear_sources();
            for (c, &p) in self.positions.iter().enumerate() {
                s.add_ion_current(p, NA, self.acc[0][c]);
                s.add_ion_current(p, K, self.acc[1][c]);
                s.add_ion_current(p, X, self.acc[2][c]);
                s.add_capacitive_current(p, self.acc[3][c]);
            }
            s.add_ion_current(self.positions[0], X, electrode);
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
    use ed_core::constants::F;

    fn small() -> TissueConfig {
        TissueConfig {
            knp: KnpConfig {
                n: [16, 16, 32],
                size: [640.0, 640.0, 1280.0],
                dt: 0.5,
                boundary: Boundary::Neumann,
                ..KnpConfig::default()
            },
            ..TissueConfig::default()
        }
    }

    /// Total amount of species `k` in the extracellular space, mM·µm³.
    fn amount(t: &Tissue, k: usize, porosity: f64) -> f64 {
        let h = t.knp.grid().h;
        t.knp.concentration(k).iter().sum::<f64>() * h[0] * h[1] * h[2] * porosity
    }

    #[test]
    fn default_cell_fits_the_column() {
        let t = Tissue::pyramidal();
        assert_eq!(t.positions().len(), 115);
        assert_eq!(t.substeps(), 40);
    }

    #[test]
    fn released_ions_match_the_membrane_currents() {
        let cfg = small();
        let mut t = Tissue::new(&cfg, Neuron::pyramidal());
        t.neuron.i_stim = 1.0;
        t.neuron.syn_rate_hz = 10_000.0;
        let por = cfg.knp.porosity;
        let m0: Vec<f64> = (0..4).map(|k| amount(&t, k, por)).collect();
        // Expected release per species from the currents actually deposited, nA·ms.
        let mut q = [0.0; 3];
        for _ in 0..40 {
            t.step(1);
            // Ionic, capacitive and electrode currents cancel over the cell.
            let net: f64 = t.acc.iter().map(|a| a.iter().sum::<f64>()).sum();
            assert!((net - t.neuron.i_stim).abs() < 1e-9, "net {net}");
            for (k, acc) in t.acc[..3].iter().enumerate() {
                q[k] += acc.iter().sum::<f64>() * t.knp.dt();
            }
        }
        // The electrode returns −i_stim as X⁻ at the soma.
        q[2] += -t.neuron.i_stim * t.time_ms();
        let z = [1.0, 1.0, -1.0];
        for k in 0..3 {
            let added = amount(&t, k, por) - m0[k];
            let expected = 1e6 / F * q[k] / z[k];
            // Round-off of summing ~1e10 mM·µm³ over the grid, or 1e-6 relative.
            let tol = 1e-12 * m0[k] + 1e-6 * expected.abs();
            assert!(
                (added - expected).abs() < tol,
                "species {k}: {added} vs {expected}"
            );
        }
        assert!((amount(&t, CA, por) - m0[CA]).abs() < 1e-12 * m0[CA]);
        assert!(t.spikes() > 0, "the stimulated cell should fire");
    }

    #[test]
    fn potassium_accumulates_at_the_soma_and_trace_is_recorded() {
        let mut t = Tissue::new(&small(), Neuron::pyramidal());
        t.neuron.syn_rate_hz = 10_000.0;
        t.step(100);
        let soma = t.knp.cell_at(t.positions()[0]).unwrap();
        let k = t.knp.concentration(K);
        let corner = k[0];
        assert!(k[soma] > 3.0 && k[soma] > corner, "{} vs {corner}", k[soma]);
        let trace = t.take_trace();
        // 100 steps × 20 neuron steps / 4 per sample, as (t, V) pairs.
        assert_eq!(trace.len(), 2 * 500);
        assert!((trace[trace.len() - 2] - 50.0).abs() < 1e-9);
        assert!(t.take_trace().is_empty());
    }

    #[test]
    fn dt_is_a_whole_number_of_neuron_steps() {
        let mut t = Tissue::new(&small(), Neuron::pyramidal());
        t.set_dt(0.11);
        assert!((t.knp.dt() - 0.1).abs() < 1e-12);
        assert_eq!(t.substeps(), 4);
        t.set_dt(0.001);
        assert_eq!(t.substeps(), 1);
        let cfg = TissueConfig {
            knp: KnpConfig {
                dt: 0.11,
                ..small().knp
            },
            ..small()
        };
        assert_eq!(Tissue::new(&cfg, Neuron::pyramidal()).substeps(), 4);
    }

    #[test]
    #[should_panic(expected = "KNP time step")]
    fn infinite_dt_is_rejected() {
        Tissue::new(&small(), Neuron::pyramidal()).set_dt(f64::INFINITY);
    }

    /// Release-mode timing of the default column:
    /// `cargo test --release -p ed-knp tissue_step_time -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn tissue_step_time() {
        let mut t = Tissue::pyramidal();
        t.neuron.syn_rate_hz = 8000.0;
        t.step(20);
        let start = std::time::Instant::now();
        t.step(100);
        let ms = start.elapsed().as_secs_f64() * 10.0;
        println!("{ms:.2} ms per 1 ms step, {} spikes", t.spikes());
    }
}
