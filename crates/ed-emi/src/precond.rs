//! Preconditioner for the KNP-EMI potential equation.

use ed_core::linalg::{DiffusionOperator, Face, LinearOperator, Multigrid, Preconditioner};

use crate::geometry::{VoxelGeometry, ECS};

/// Multigrid plus an additive coarse correction for the intracellular "cable" modes.
///
/// Membranes conduct only weakly (C_M/Δt), so the potential of each cell relative to the
/// extracellular space is a set of slow modes that geometric multigrid cannot represent
/// once coarse cells straddle a membrane. The coarse space has one vector per cell and
/// x-layer (the indicator of the cell's voxels in that layer); restricted to it, the
/// operator is tridiagonal per cell and is solved exactly.
pub struct CablePreconditioner {
    mg: Multigrid,
    /// Coarse index per voxel (`usize::MAX` outside cells).
    dof: Vec<usize>,
    /// First coarse index of each cell's chain, plus the end.
    chain: Vec<usize>,
    diag: Vec<f64>,
    /// Coupling of dof `s` to `s + 1` (0 between chains).
    off: Vec<f64>,
    rhs: Vec<f64>,
    scratch: Vec<f64>,
    full_diag: Vec<f64>,
}

impl CablePreconditioner {
    pub fn new(op: &DiffusionOperator, geo: &VoxelGeometry) -> Self {
        let g = geo.grid;
        let ntags = geo.voxel_counts().len();
        let (mut lo, mut hi) = (vec![usize::MAX; ntags], vec![0usize; ntags]);
        g.for_each_cell(|i, _, _, v| {
            let t = geo.tags[v] as usize;
            lo[t] = lo[t].min(i);
            hi[t] = hi[t].max(i + 1);
        });
        let mut chain = vec![0];
        for t in 0..ntags {
            let len = if t == ECS as usize || lo[t] == usize::MAX {
                0
            } else {
                hi[t] - lo[t]
            };
            chain.push(chain[t] + len);
        }
        let n = *chain.last().expect("chain end");
        let dof = (0..g.len())
            .map(|v| {
                let t = geo.tags[v] as usize;
                if t == ECS as usize {
                    usize::MAX
                } else {
                    chain[t] + v % g.n[0] - lo[t]
                }
            })
            .collect();
        let mut p = Self {
            mg: Multigrid::new(op),
            dof,
            chain,
            diag: vec![0.0; n],
            off: vec![0.0; n],
            rhs: vec![0.0; n],
            scratch: vec![0.0; n],
            full_diag: vec![0.0; g.len()],
        };
        p.assemble(op);
        p
    }

    /// Re-derives the multigrid levels and the coarse matrix after `op` changed.
    pub fn update(&mut self, op: &DiffusionOperator) {
        self.mg.update(op);
        self.assemble(op);
    }

    fn assemble(&mut self, op: &DiffusionOperator) {
        let g = *op.grid();
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        op.diagonal(&mut self.full_diag);
        self.diag.fill(0.0);
        self.off.fill(0.0);
        let (dof, diag, off, full) = (&self.dof, &mut self.diag, &mut self.off, &self.full_diag);
        g.for_each_cell(|i, j, k, v| {
            let s = dof[v];
            if s == usize::MAX {
                return;
            }
            diag[s] += full[v];
            let pos = [i, j, k];
            for a in 0..3 {
                if pos[a] + 1 >= g.n[a] {
                    continue;
                }
                let u = v + stride[a];
                let t = dof[u];
                if t == usize::MAX {
                    continue;
                }
                let w = op.face_weight(Face { axis: a, cell: v });
                if t == s {
                    diag[s] -= 2.0 * w;
                } else if t == s + 1 {
                    off[s] -= w;
                }
            }
        });
    }
}

impl Preconditioner for CablePreconditioner {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        self.mg.apply(r, z);
        self.rhs.fill(0.0);
        for (&s, &rv) in self.dof.iter().zip(r) {
            if s != usize::MAX {
                self.rhs[s] += rv;
            }
        }
        // Thomas algorithm per chain (the matrix is SPD, so no pivoting is needed).
        for w in self.chain.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a == b {
                continue;
            }
            let (d, o, y, c) = (&self.diag, &self.off, &mut self.rhs, &mut self.scratch);
            c[a] = o[a] / d[a];
            y[a] /= d[a];
            for s in a + 1..b {
                let m = d[s] - o[s - 1] * c[s - 1];
                c[s] = o[s] / m;
                y[s] = (y[s] - o[s - 1] * y[s - 1]) / m;
            }
            for s in (a..b - 1).rev() {
                y[s] -= c[s] * y[s + 1];
            }
        }
        for (zv, &s) in z.iter_mut().zip(&self.dof) {
            if s != usize::MAX {
                *zv += self.rhs[s];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::AxonBundle;
    use ed_core::linalg::{BoundaryConditions, CgOptions, NullSpace, Pcg};

    #[test]
    fn cable_space_speeds_up_membrane_problems() {
        let (geo, _) = AxonBundle::default().build();
        let faces: Vec<(Face, f64)> = geo
            .membrane_faces()
            .iter()
            .map(|f| (f.face, 1e-4))
            .collect();
        let mut op = DiffusionOperator::uniform(geo.grid, BoundaryConditions::neumann(), 1.0);
        op.set_face_sigma(&faces);
        let b: Vec<f64> = (0..op.len()).map(|i| (i as f64 * 0.37).sin()).collect();
        let opts = CgOptions {
            rtol: 1e-8,
            null_space: NullSpace::Constant,
            ..Default::default()
        };
        let mut cg = Pcg::new(op.len());
        let mut run = |m: &mut dyn Preconditioner| {
            let mut x = vec![0.0; op.len()];
            let s = cg.solve(&op, m, &b, &mut x, &opts);
            assert!(s.converged, "{s:?}");
            s.iterations
        };
        let plain = run(&mut Multigrid::new(&op));
        let mut cable = CablePreconditioner::new(&op, &geo);
        let with_cable = run(&mut cable);
        cable.update(&op);
        assert_eq!(run(&mut cable), with_cable);
        assert!(
            2 * with_cable < plain,
            "cable {with_cable} vs multigrid {plain}"
        );
    }
}
