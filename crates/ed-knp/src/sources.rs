//! Point sources: transmembrane currents of neuron compartments, binned into grid cells.

use crate::KnpSolver;
use ed_core::constants::F;

/// Converts a current in nA carried by charge `z` into an amount rate in mM·µm³/ms.
fn na_to_rate(i_na: f64, z: f64) -> f64 {
    // 1 nA = 1e-9 C/s; 1 mM = 1 mol/m³, so 1 mol = 1e18 mM·µm³; 1 s = 1e3 ms.
    1e6 * i_na / (z * F)
}

impl KnpSolver {
    /// Index of the cell containing `pos` (µm), or `None` outside the domain.
    pub fn cell_at(&self, pos: [f64; 3]) -> Option<usize> {
        let g = &self.model.grid;
        let mut ijk = [0usize; 3];
        for a in 0..3 {
            let u = pos[a] / g.h[a];
            if !(u >= 0.0 && u < g.n[a] as f64) {
                return None;
            }
            ijk[a] = u as usize;
        }
        Some(g.index(ijk[0], ijk[1], ijk[2]))
    }

    /// Removes all point sources. Only the cells that received one are touched.
    pub fn clear_sources(&mut self) {
        for &cell in &self.source_cells {
            self.src.iter_mut().for_each(|s| s[cell] = 0.0);
            self.cap[cell] = 0.0;
            self.has_source[cell] = false;
        }
        self.source_cells.clear();
    }

    /// Adds a source of species `k` at `pos`, as an amount rate in mM·µm³/ms (1e-18 mol/ms)
    /// released into the tissue. Negative rates are sinks. Returns false (and does nothing)
    /// outside the domain.
    pub fn add_ion_source(&mut self, pos: [f64; 3], k: usize, rate: f64) -> bool {
        let Some(cell) = self.cell_at(pos) else {
            return false;
        };
        self.src[k][cell] += rate * self.source_scale;
        self.mark_source(cell);
        true
    }

    /// Adds the transmembrane current (nA) carried by species `k` at `pos`, positive outward
    /// as a conventional current: an outward current releases cations but takes up anions.
    pub fn add_ion_current(&mut self, pos: [f64; 3], k: usize, i_na: f64) -> bool {
        let z = self.model.z[k];
        assert!(z != 0.0, "a neutral species carries no current");
        self.add_ion_source(pos, k, na_to_rate(i_na, z))
    }

    /// Adds a capacitive current (nA) leaving the neuron at `pos`.
    pub fn add_capacitive_current(&mut self, pos: [f64; 3], i_na: f64) -> bool {
        let Some(cell) = self.cell_at(pos) else {
            return false;
        };
        self.cap[cell] += na_to_rate(i_na, 1.0) * self.source_scale;
        self.mark_source(cell);
        true
    }

    fn mark_source(&mut self, cell: usize) {
        if !self.has_source[cell] {
            self.has_source[cell] = true;
            self.source_cells.push(cell);
        }
    }
}
