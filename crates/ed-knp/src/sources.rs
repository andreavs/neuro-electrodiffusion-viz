//! Point sources: transmembrane currents of neuron compartments, binned into grid cells.

use crate::KnpSolver;

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

    fn mark_source(&mut self, cell: usize) {
        if !self.has_source[cell] {
            self.has_source[cell] = true;
            self.source_cells.push(cell);
        }
    }
}
