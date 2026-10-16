//! Bindings for `ed-knp`: a pyramidal cell releasing ions into a column of tissue
//! (Publication 1), either the simplified cell or the detailed Hay et al. cell.
//!
//! [`KnpSim`] owns an [`ed_knp::tissue::Tissue`]. Positions are in grid coordinates (µm,
//! the box spans `[0, size]` on each axis); fields are flat, x fastest.

use ed_core::Simulation;
use ed_knp::cell::Cell;
use ed_knp::hay::{HayMorphology, HayNeuron, HayParams};
use ed_knp::neuron::{Morphology, Neuron, NeuronParams};
use ed_knp::tissue::{Tissue, TissueConfig, CA, K, NA, X};
use wasm_bindgen::prelude::*;

/// Highest synaptic rate accepted, Hz (the soma blocks above ~30 kHz).
const MAX_SYN_RATE_HZ: f64 = 50_000.0;
/// Largest somatic current accepted, nA.
const MAX_STIM_NA: f64 = 10.0;
/// Accepted KNP time steps, ms.
const DT_RANGE: (f64, f64) = (0.025, 10.0);
/// Quantities reported at the probes, in this order.
const PROBE_FIELDS: [&str; 5] = ["k", "na", "ca", "x", "phi"];

#[wasm_bindgen]
pub struct KnpSim {
    tissue: Tissue<Box<dyn Cell>>,
}

#[wasm_bindgen]
impl KnpSim {
    /// `cell` is "simple" (the default column and simplified cell) or "hay" (the Hay et
    /// al. cell in a taller column); `seed` drives the synaptic input.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, cell: &str) -> Result<KnpSim, JsError> {
        let seed = u64::from(seed);
        let (cfg, neuron): (_, Box<dyn Cell>) = match cell {
            "simple" => {
                let params = NeuronParams {
                    seed,
                    ..NeuronParams::default()
                };
                let n = Neuron::new(Morphology::pyramidal(), params);
                (TissueConfig::default(), Box::new(n))
            }
            "hay" => {
                let params = HayParams {
                    seed,
                    ..HayParams::default()
                };
                let n = HayNeuron::new(HayMorphology::cell1(), params);
                (TissueConfig::hay(), Box::new(n))
            }
            _ => return Err(JsError::new(&format!("unknown cell {cell:?}"))),
        };
        Ok(Self {
            tissue: Tissue::new(&cfg, neuron),
        })
    }

    /// Total excitatory synaptic rate over the dendrites, Hz.
    pub fn set_syn_rate(&mut self, hz: f64) -> Result<(), JsError> {
        if !(0.0..=MAX_SYN_RATE_HZ).contains(&hz) {
            return Err(JsError::new(&format!(
                "synaptic rate must be within 0–{MAX_SYN_RATE_HZ} Hz"
            )));
        }
        self.tissue.neuron.set_syn_rate(hz);
        Ok(())
    }
    /// Current injected into the soma, nA (depolarising positive).
    pub fn set_stim(&mut self, i_na: f64) -> Result<(), JsError> {
        if !(-MAX_STIM_NA..=MAX_STIM_NA).contains(&i_na) {
            return Err(JsError::new(&format!(
                "somatic current must be within ±{MAX_STIM_NA} nA"
            )));
        }
        self.tissue.neuron.set_stim(i_na);
        Ok(())
    }
    /// KNP time step, ms (rounded to whole neuron steps of 0.025 ms).
    pub fn set_dt(&mut self, dt: f64) -> Result<(), JsError> {
        if !(DT_RANGE.0..=DT_RANGE.1).contains(&dt) {
            return Err(JsError::new(&format!(
                "time step must be within {}–{} ms",
                DT_RANGE.0, DT_RANGE.1
            )));
        }
        self.tissue.set_dt(dt);
        Ok(())
    }

    pub fn dt(&self) -> f64 {
        self.tissue.knp.dt()
    }
    /// Takes `n` KNP steps.
    pub fn step(&mut self, n: usize) {
        self.tissue.step(n);
    }

    pub fn time_ms(&self) -> f64 {
        self.tissue.time_ms()
    }
    /// Spikes at the soma so far.
    pub fn spikes(&self) -> f64 {
        self.tissue.spikes() as f64
    }
    /// Cells per axis `[nx, ny, nz]`.
    pub fn shape(&self) -> Vec<u32> {
        self.tissue.knp.grid().n.iter().map(|&n| n as u32).collect()
    }

    /// Box size per axis, µm.
    pub fn size(&self) -> Vec<f64> {
        let g = self.tissue.knp.grid();
        (0..3).map(|a| g.n[a] as f64 * g.h[a]).collect()
    }

    /// A whole field: "k", "na", "ca", "x" (mM) or "phi" (mV).
    pub fn field(&self, name: &str) -> Result<Vec<f32>, JsError> {
        Ok(self.values(name)?.iter().map(|&v| v as f32).collect())
    }

    /// Membrane potential per compartment, mV.
    pub fn v(&self) -> Vec<f32> {
        self.tissue.neuron.v().iter().map(|&v| v as f32).collect()
    }

    /// Per compartment: start xyz, end xyz (grid coordinates, µm) and diameter (µm).
    pub fn compartments(&self) -> Vec<f32> {
        let o = self.tissue.soma();
        let mut out = Vec::with_capacity(7 * self.tissue.neuron.len());
        for c in &self.tissue.neuron.morphology().compartments {
            out.extend((0..3).map(|a| (c.start[a] + o[a]) as f32));
            out.extend((0..3).map(|a| (c.end[a] + o[a]) as f32));
            out.push(c.diam as f32);
        }
        out
    }

    /// Somatic (t ms, V mV) pairs recorded since the last call.
    pub fn take_trace(&mut self) -> Vec<f64> {
        self.tissue.take_trace()
    }

    /// For each probe position (flat xyz, µm): [K⁺], [Na⁺], [Ca²⁺], [X⁻] and φ in its
    /// cell, or NaN outside the box.
    pub fn probe(&self, xyz: &[f64]) -> Vec<f32> {
        let knp = &self.tissue.knp;
        let mut out = Vec::with_capacity(xyz.len() / 3 * PROBE_FIELDS.len());
        for p in xyz.chunks_exact(3) {
            let cell = knp.cell_at([p[0], p[1], p[2]]);
            for name in PROBE_FIELDS {
                let v = match (cell, self.values(name)) {
                    (Some(c), Ok(f)) => f[c] as f32,
                    _ => f32::NAN,
                };
                out.push(v);
            }
        }
        out
    }
}

impl KnpSim {
    fn values(&self, name: &str) -> Result<&[f64], JsError> {
        let knp = &self.tissue.knp;
        Ok(match name {
            "k" => knp.concentration(K),
            "na" => knp.concentration(NA),
            "ca" => knp.concentration(CA),
            "x" => knp.concentration(X),
            "phi" => knp.potential(),
            _ => return Err(JsError::new(&format!("unknown field {name:?}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hay_cell_runs_in_its_column() {
        let mut s = KnpSim::new(1, "hay").unwrap();
        assert_eq!(s.shape(), vec![32, 32, 80]);
        assert_eq!(s.compartments().len(), 7 * 642);
        s.set_stim(2.0).unwrap();
        s.step(20);
        assert!(s.spikes() > 0.0);
    }

    #[test]
    fn fields_and_probes_have_the_grid_layout() {
        let mut s = KnpSim::new(1, "simple").unwrap();
        s.set_syn_rate(10_000.0).unwrap();
        s.step(5);
        assert!((s.time_ms() - 5.0).abs() < 1e-9);
        let n = s.shape();
        let len = (n[0] * n[1] * n[2]) as usize;
        for f in PROBE_FIELDS {
            assert_eq!(s.field(f).unwrap().len(), len);
        }
        assert_eq!(s.compartments().len(), 7 * s.v().len());
        let soma = s.tissue.positions()[0];
        let p = s.probe(&[soma[0], soma[1], soma[2], -1.0, 0.0, 0.0]);
        assert_eq!(p.len(), 10);
        let cell = s.tissue.knp.cell_at(soma).unwrap();
        assert_eq!(p[0], s.field("k").unwrap()[cell]);
        assert_eq!(p[4], s.field("phi").unwrap()[cell]);
        assert!(p[5..].iter().all(|v| v.is_nan()));
        assert_eq!(s.take_trace().len(), 2 * 5 * 10);
    }
}
