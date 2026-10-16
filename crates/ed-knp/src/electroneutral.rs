//! Electroneutral projection: the KNP potential and the drift it drives.

use crate::Model;
use ed_core::linalg::{CgOptions, CgStats, DiffusionOperator, Multigrid, NullSpace, Pcg};

/// The potential operator (and its multigrid hierarchy) is rebuilt only when κ has moved by
/// more than this fraction in some cell since the last rebuild. The drift uses the operator's
/// conductivity, so the charge balance is exact either way; the lag only perturbs how the
/// current is shared between faces, by at most this fraction.
const KAPPA_REFRESH: f64 = 1e-2;

/// Solves `−∇·σ∇φ = (Σ z c* − ρ)/Δt` with `σ ≈ κ = Σ_k D̃_k z_k² c_k / ψ` and applies the
/// drift `c ← c* + Δt ∇·(D̃ z c* ∇φ / ψ)`, so that `Σ z c` lands on the target `ρ`.
pub(crate) struct Projection {
    /// Target charge density Σ z c (mM): its initial value minus the integrated capacitive
    /// sink. Projecting onto the target rather than onto Σ z cⁿ keeps solver tolerance errors
    /// from accumulating.
    charge: Vec<f64>,
    charge_stale: bool,
    charge_tol: f64,
    phi: Vec<f64>,
    kappa: Vec<f64>,
    rhs: Vec<f64>,
    /// Potential operator `−∇·σ∇`, see [`KAPPA_REFRESH`].
    op: DiffusionOperator,
    mg: Multigrid,
    pcg: Pcg,
    opts: CgOptions,
    last_solve: CgStats,
    /// Drift weight of the upper face of each cell along each axis, of the Dirichlet faces,
    /// and the next concentration per species.
    face_weight: [Vec<f64>; 3],
    boundary: Vec<f64>,
    next: Vec<Vec<f64>>,
}

impl Projection {
    pub fn new(m: &Model, charge_tol: f64, dt: f64) -> Self {
        let len = m.grid.len();
        let op = DiffusionOperator::uniform(m.grid, m.bc, 1.0);
        let null_space = if m.bc.is_pure_neumann(&m.grid) {
            NullSpace::Constant
        } else {
            NullSpace::None
        };
        let mut s = Self {
            charge: vec![0.0; len],
            charge_stale: true,
            charge_tol,
            phi: vec![0.0; len],
            kappa: vec![0.0; len],
            rhs: vec![0.0; len],
            mg: Multigrid::new(&op),
            op,
            pcg: Pcg::new(len),
            opts: CgOptions {
                rtol: 0.0,
                atol: 0.0,
                max_iter: 500,
                null_space,
            },
            last_solve: CgStats {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
            face_weight: [vec![0.0; len], vec![0.0; len], vec![0.0; len]],
            boundary: vec![0.0; len],
            next: vec![vec![0.0; len]; m.d.len()],
        };
        s.set_dt(m, dt);
        s
    }

    pub fn set_dt(&mut self, m: &Model, dt: f64) {
        // The residual is a charge rate (mM/ms), measured in the 2-norm over all cells.
        self.opts.atol = self.charge_tol * (m.grid.len() as f64).sqrt() / dt;
    }

    /// Takes the current Σ z c as the new target, after the concentrations were edited.
    pub fn reset_charge(&mut self) {
        self.charge_stale = true;
    }

    pub fn potential(&self) -> &[f64] {
        &self.phi
    }

    pub fn last_solve(&self) -> CgStats {
        self.last_solve
    }

    /// The target charge before a step, from `c` if it was reset.
    pub fn prepare(&mut self, m: &Model, c: &[Vec<f64>]) {
        if self.charge_stale {
            self.charge.fill(0.0);
            for (ck, z) in c.iter().zip(&m.z) {
                self.charge
                    .iter_mut()
                    .zip(ck)
                    .for_each(|(q, c)| *q += z * c);
            }
            self.charge_stale = false;
        }
    }

    /// Projects `c` (after diffusion) onto the target charge, which first moves by
    /// `−Δt q̃_cap` (`cap` in mM/ms).
    pub fn step(&mut self, m: &Model, dt: f64, c: &mut [Vec<f64>], cap: &[f64]) {
        let (len, psi) = (m.grid.len(), m.psi);
        let (kappa, rhs) = (&mut self.kappa, &mut self.rhs);
        for ((rho, q), b) in self.charge.iter_mut().zip(cap).zip(rhs.iter_mut()) {
            *rho -= dt * q;
            *b = -*rho / dt;
        }
        kappa.fill(0.0);
        for ((ck, d), z) in c.iter().zip(&m.d).zip(&m.z) {
            let (sk, qk) = (d * z * z / psi, z / dt);
            for ((kap, b), c) in kappa.iter_mut().zip(rhs.iter_mut()).zip(ck) {
                *kap += sk * c;
                *b += qk * c;
            }
        }
        let sigma = self.op.sigma().iter();
        if sigma
            .zip(kappa.iter())
            .any(|(s, k)| (s - k).abs() > KAPPA_REFRESH * k)
        {
            self.op.update_sigma(kappa);
            self.mg.update(&self.op);
        }
        self.last_solve = self
            .pcg
            .solve(&self.op, &mut self.mg, rhs, &mut self.phi, &self.opts);
        debug_assert!(
            self.last_solve.converged,
            "potential solve: {:?}",
            self.last_solve
        );

        // Each species carries a share of the face current σ_f Δφ / h² proportional to D̃ z² c
        // at the face, so Σ z (drift) equals ∇·σ∇φ of the operator exactly. σ_f is the
        // harmonic mean of the operator's σ, and Σ_k D̃ z² c_face = ψ κ̄, κ̄ = (κ_a + κ_b)/2.
        let (g, phi, kappa, sigma) = (m.grid, &self.phi, &self.kappa, self.op.sigma());
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        for (a, w) in self.face_weight.iter_mut().enumerate() {
            // w_a[cell] = σ_f Δφ / (2 h² ψ κ̄) for the face between `cell` and `cell + s`, and
            // 0 for faces that wrap around a row or plane (and on the last layer).
            w.fill(0.0);
            let s = stride[a];
            if g.n[a] == 1 {
                continue;
            }
            let scale = 2.0 / (g.h[a] * g.h[a] * psi);
            let k_pairs = kappa[..len - s].iter().zip(&kappa[s..]);
            let s_pairs = sigma[..len - s].iter().zip(&sigma[s..]);
            let p_pairs = phi[..len - s].iter().zip(&phi[s..]);
            let faces = w.iter_mut().zip(k_pairs).zip(s_pairs).zip(p_pairs);
            for (((w, (ka, kb)), (sa, sb)), (pa, pb)) in faces {
                let (ks, ss) = (ka + kb, sa + sb);
                if ks > 0.0 && ss > 0.0 {
                    *w = scale * sa * sb / (ss * ks) * (pb - pa);
                }
            }
            let [nx, ny, _] = g.n;
            match a {
                0 => w.iter_mut().skip(nx - 1).step_by(nx).for_each(|w| *w = 0.0),
                1 => w
                    .chunks_mut(nx * ny)
                    .for_each(|plane| plane[(ny - 1) * nx..].fill(0.0)),
                _ => {}
            }
        }
        // Dirichlet faces: ghost φ = 0 and face values = cell values, so the current σ φ is
        // shared in proportion to D̃ z² c / (ψ κ).
        let wb = &mut self.boundary;
        let bw = m.boundary_weight.iter().zip(sigma).zip(kappa);
        for ((w, ((b, s), k)), p) in wb.iter_mut().zip(bw).zip(phi) {
            *w = if *k > 0.0 { b * s / (k * psi) * p } else { 0.0 };
        }

        let [w0, w1, w2] = &self.face_weight;
        let (s1, s2) = (stride[1], stride[2]);
        for (k, ck) in c.iter_mut().enumerate() {
            let dz = dt * m.d[k] * m.z[k];
            if dz == 0.0 {
                continue;
            }
            let out = &mut self.next[k];
            // Cells away from the first and last z-plane have all six neighbours in range;
            // equal-length slices let the compiler drop bounds checks and vectorise.
            let lo = s2.min(len);
            let hi = len.saturating_sub(s2).max(lo);
            let n = hi - lo;
            let at = |v, off| window(v, lo, off, n);
            let (s1i, s2i) = (s1 as isize, s2 as isize);
            if n > 0 {
                let x = at(ck, 0);
                let (xp0, xm0, xp1, xm1) = (at(ck, 1), at(ck, -1), at(ck, s1i), at(ck, -s1i));
                let (xp2, xm2) = (at(ck, s2i), at(ck, -s2i));
                let (w0c, w0m, w1c, w1m) = (at(w0, 0), at(w0, -1), at(w1, 0), at(w1, -s1i));
                let (w2c, w2m, wbc) = (at(w2, 0), at(w2, -s2i), at(wb, 0));
                for (i, o) in out[lo..hi].iter_mut().enumerate() {
                    let xi = x[i];
                    let acc = w0c[i] * (xi + xp0[i]) - w0m[i] * (xm0[i] + xi)
                        + w1c[i] * (xi + xp1[i])
                        - w1m[i] * (xm1[i] + xi)
                        + w2c[i] * (xi + xp2[i])
                        - w2m[i] * (xm2[i] + xi)
                        - wbc[i] * xi;
                    *o = xi + dz * acc;
                }
            }
            for cell in (0..lo).chain(hi..len) {
                let x = ck[cell];
                let mut acc = -wb[cell] * x;
                for (w, s) in [(w0, 1), (w1, s1), (w2, s2)] {
                    if cell + s < len {
                        acc += w[cell] * (x + ck[cell + s]);
                    }
                    if cell >= s {
                        acc -= w[cell - s] * (ck[cell - s] + x);
                    }
                }
                out[cell] = x + dz * acc;
            }
            std::mem::swap(ck, out);
        }
    }
}

/// `v[start + off .. start + off + n]`.
fn window(v: &[f64], start: usize, off: isize, n: usize) -> &[f64] {
    &v[start.wrapping_add_signed(off)..][..n]
}
