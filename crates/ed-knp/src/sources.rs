//! Point sources: transmembrane currents of neuron compartments, binned into
//! the grid cell that contains each position.

use crate::KnpSim;
use ed_core::constants::F;

/// amol/ms carried by 1 nA of a monovalent ion: 1 nA = 10⁻¹² C/ms, 1 amol = 10⁻¹⁸ mol.
const AMOL_PER_MS_PER_NA: f64 = 1e6 / F;

impl KnpSim {
    /// Removes all sources. Sources otherwise persist from step to step.
    pub fn clear_sources(&mut self) {
        self.src.iter_mut().for_each(|s| s.fill(0.0));
        self.cap.fill(0.0);
    }

    /// Index of the cell containing `pos` (µm), or `None` outside the box.
    pub fn cell_at(&self, pos: [f64; 3]) -> Option<usize> {
        let g = &self.grid;
        let mut ijk = [0; 3];
        for a in 0..3 {
            let x = pos[a] / g.h[a];
            if !(0.0..g.n[a] as f64).contains(&x) {
                return None;
            }
            ijk[a] = x as usize;
        }
        Some(g.index(ijk[0], ijk[1], ijk[2]))
    }

    /// Adds a source of species `k` at `pos` (µm) releasing `rate` amol/ms
    /// (negative for uptake). A charged species carries a current; with
    /// [`OuterBoundary::NoFlux`](crate::OuterBoundary::NoFlux) the total over all
    /// sources must balance. Returns `false`, adding nothing, if `pos` lies
    /// outside the box.
    pub fn add_species_source(&mut self, pos: [f64; 3], k: usize, rate: f64) -> bool {
        let Some(cell) = self.cell_at(pos) else {
            return false;
        };
        self.src[k][cell] += rate * self.inv_cell_volume();
        true
    }

    /// Adds a neuron compartment at `pos` (µm) with per-species ionic currents
    /// and a capacitive current, all in nA and positive out of the neuron.
    /// Currents of neutral species are ignored. With
    /// [`OuterBoundary::NoFlux`](crate::OuterBoundary::NoFlux) the summed membrane
    /// current over all sources must be zero. Returns `false`, adding
    /// nothing, if `pos` lies outside the box.
    pub fn add_current_source(&mut self, pos: [f64; 3], ionic_na: &[f64], cap_na: f64) -> bool {
        assert_eq!(ionic_na.len(), self.c.len(), "one current per species");
        let Some(cell) = self.cell_at(pos) else {
            return false;
        };
        let scale = AMOL_PER_MS_PER_NA * self.inv_cell_volume();
        for (k, &i) in ionic_na.iter().enumerate() {
            if self.z[k] != 0.0 {
                self.src[k][cell] += i * scale / self.z[k];
            }
        }
        self.cap[cell] += cap_na * scale;
        true
    }

    /// `1 / (α V_cell)`: converts amol/ms in one cell to mM/ms of extracellular fluid.
    fn inv_cell_volume(&self) -> f64 {
        let h = self.grid.h;
        1.0 / (h[0] * h[1] * h[2] * self.params.volume_fraction)
    }
}
