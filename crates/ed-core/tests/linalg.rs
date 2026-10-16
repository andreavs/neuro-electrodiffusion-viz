//! Convergence and solver tests for `ed_core::linalg` on manufactured solutions.

use ed_core::linalg::{
    BoundaryConditions, CgOptions, DiffusionOperator, Grid, Jacobi, LinearOperator, Multigrid,
    NullSpace, Pcg, Preconditioner,
};
use std::f64::consts::PI;

/// A manufactured problem `-∇·(σ∇u) = f` with exact solution `u`.
struct Manufactured {
    u: fn([f64; 3]) -> f64,
    sigma: fn([f64; 3]) -> f64,
    f: fn([f64; 3]) -> f64,
}

// 2-D, Dirichlet: u = sin(2x + 1) cos(y), σ = 1 + x/2 + y/4 on [0, 1]².
// -∇·(σ∇u) = -(σ Δu + σ_x u_x + σ_y u_y), Δu = -5u.
const DIRICHLET_2D: Manufactured = Manufactured {
    u: |p| (2.0 * p[0] + 1.0).sin() * p[1].cos(),
    sigma: |p| 1.0 + 0.5 * p[0] + 0.25 * p[1],
    f: |p| {
        let (x, y) = (p[0], p[1]);
        let u = (2.0 * x + 1.0).sin() * y.cos();
        let ux = 2.0 * (2.0 * x + 1.0).cos() * y.cos();
        let uy = -(2.0 * x + 1.0).sin() * y.sin();
        let s = 1.0 + 0.5 * x + 0.25 * y;
        -(s * (-5.0 * u) + 0.5 * ux + 0.25 * uy)
    },
};

// 3-D, Dirichlet: u = exp(x) sin(y + z/2), σ = 2 + x y z, on [0, 1]³.
// Δu = u (1 - 1 - 1/4) = -u/4.
const DIRICHLET_3D: Manufactured = Manufactured {
    u: |p| p[0].exp() * (p[1] + 0.5 * p[2]).sin(),
    sigma: |p| 2.0 + p[0] * p[1] * p[2],
    f: |p| {
        let (x, y, z) = (p[0], p[1], p[2]);
        let e = x.exp();
        let (s, c) = (y + 0.5 * z).sin_cos();
        let u = e * s;
        let (ux, uy, uz) = (u, e * c, 0.5 * e * c);
        let sig = 2.0 + x * y * z;
        -(sig * (-0.25 * u) + y * z * ux + x * z * uy + x * y * uz)
    },
};

// 3-D, Neumann: u = cos(πx) cos(πy) cos(πz) has zero normal derivative and zero mean on
// [0, 1]³. σ = 1 + x/2.
const NEUMANN_3D: Manufactured = Manufactured {
    u: |p| (PI * p[0]).cos() * (PI * p[1]).cos() * (PI * p[2]).cos(),
    sigma: |p| 1.0 + 0.5 * p[0],
    f: |p| {
        let (cx, cy, cz) = ((PI * p[0]).cos(), (PI * p[1]).cos(), (PI * p[2]).cos());
        let u = cx * cy * cz;
        let ux = -PI * (PI * p[0]).sin() * cy * cz;
        -((1.0 + 0.5 * p[0]) * (-3.0 * PI * PI * u) + 0.5 * ux)
    },
};

fn setup(
    grid: Grid,
    bc: BoundaryConditions,
    m: &Manufactured,
) -> (DiffusionOperator, Vec<f64>, Vec<f64>) {
    let mut sigma = vec![0.0; grid.len()];
    let mut b = vec![0.0; grid.len()];
    let mut exact = vec![0.0; grid.len()];
    grid.for_each_cell(|i, j, k, c| {
        let p = grid.cell_center(i, j, k);
        sigma[c] = (m.sigma)(p);
        b[c] = (m.f)(p);
        exact[c] = (m.u)(p);
    });
    let op = DiffusionOperator::new(grid, bc, &sigma);
    op.add_dirichlet_rhs(&mut b, |_, _, p| (m.u)(p));
    (op, b, exact)
}

fn solve_mg(op: &DiffusionOperator, b: &[f64], null_space: NullSpace) -> Vec<f64> {
    let mut x = vec![0.0; op.len()];
    let opts = CgOptions {
        rtol: 1e-11,
        null_space,
        ..Default::default()
    };
    let stats = Pcg::new(op.len()).solve(op, &mut Multigrid::new(op), b, &mut x, &opts);
    assert!(stats.converged, "{stats:?}");
    x
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Every refinement must gain at least order 1.7 (coarse grids are pre-asymptotic because
/// the Dirichlet ghost-cell closure is first order locally) and the finest pair at least 1.85.
fn assert_second_order(errors: &[f64]) {
    let orders: Vec<f64> = errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
    let ok = orders.iter().all(|&o| o > 1.7) && *orders.last().unwrap() > 1.85;
    assert!(ok, "observed orders {orders:.3?}, errors {errors:?}");
}

#[test]
fn dirichlet_2d_converges_at_second_order() {
    let errors: Vec<f64> = [16, 32, 64, 128]
        .iter()
        .map(|&n| {
            let h = 1.0 / n as f64;
            let (op, b, exact) = setup(
                Grid::new_2d(n, n, h, h),
                BoundaryConditions::dirichlet(),
                &DIRICHLET_2D,
            );
            max_err(&solve_mg(&op, &b, NullSpace::None), &exact)
        })
        .collect();
    assert_second_order(&errors);
}

#[test]
fn dirichlet_3d_converges_at_second_order() {
    let errors: Vec<f64> = [16, 32, 64]
        .iter()
        .map(|&n| {
            let h = 1.0 / n as f64;
            let (op, b, exact) = setup(
                Grid::new_3d([n; 3], [h; 3]),
                BoundaryConditions::dirichlet(),
                &DIRICHLET_3D,
            );
            max_err(&solve_mg(&op, &b, NullSpace::None), &exact)
        })
        .collect();
    assert_second_order(&errors);
}

#[test]
fn neumann_solution_has_zero_mean_and_correct_gradients() {
    let mut errors = Vec::new();
    let mut grad_errors = Vec::new();
    for n in [8, 16, 32] {
        let h = 1.0 / n as f64;
        let grid = Grid::new_3d([n; 3], [h; 3]);
        let (op, b, exact) = setup(grid, BoundaryConditions::neumann(), &NEUMANN_3D);
        let x = solve_mg(&op, &b, NullSpace::Constant);

        let mean = x.iter().sum::<f64>() / x.len() as f64;
        assert!(mean.abs() < 1e-12, "mean {mean}");
        errors.push(max_err(&x, &exact));

        // x-gradient at interior faces vs the analytic derivative at the face centre.
        let mut g_err: f64 = 0.0;
        grid.for_each_cell(|i, j, k, c| {
            if i + 1 < n {
                let mut p = grid.cell_center(i, j, k);
                p[0] += 0.5 * h;
                let exact_ux = -PI * (PI * p[0]).sin() * (PI * p[1]).cos() * (PI * p[2]).cos();
                g_err = g_err.max(((x[c + 1] - x[c]) / h - exact_ux).abs());
            }
        });
        grad_errors.push(g_err);
    }
    assert_second_order(&errors);
    assert_second_order(&grad_errors);
}

#[test]
fn multigrid_and_jacobi_agree_and_multigrid_needs_fewer_iterations() {
    let n = 32;
    let h = 1.0 / n as f64;
    let (op, b, _) = setup(
        Grid::new_3d([n, n, n / 2], [h; 3]),
        BoundaryConditions::neumann(),
        &NEUMANN_3D,
    );
    let opts = CgOptions {
        rtol: 1e-10,
        null_space: NullSpace::Constant,
        ..Default::default()
    };
    let mut cg = Pcg::new(op.len());
    let run = |cg: &mut Pcg, m: &mut dyn Preconditioner| {
        let mut x = vec![0.0; op.len()];
        let stats = cg.solve(&op, m, &b, &mut x, &opts);
        assert!(stats.converged, "{stats:?}");
        (x, stats.iterations)
    };
    let (xj, it_j) = run(&mut cg, &mut Jacobi::new(&op));
    let (xm, it_m) = run(&mut cg, &mut Multigrid::new(&op));
    assert!(max_err(&xj, &xm) < 1e-7);
    assert!(
        it_m * 4 < it_j,
        "multigrid {it_m} vs Jacobi {it_j} iterations"
    );
}

#[test]
fn multigrid_update_tracks_new_coefficients() {
    let n = 16;
    let h = 1.0 / n as f64;
    let (mut op, b, _) = setup(
        Grid::new_2d(n, n, h, h),
        BoundaryConditions::dirichlet(),
        &DIRICHLET_2D,
    );
    let mut mg = Multigrid::new(&op);
    let scaled: Vec<f64> = op.sigma().iter().map(|s| 3.0 * s).collect();
    op.update_sigma(&scaled);
    mg.update(&op);
    let fresh = Multigrid::new(&op);
    let (mut z1, mut z2) = (vec![0.0; op.len()], vec![0.0; op.len()]);
    mg.apply(&b, &mut z1);
    { fresh }.apply(&b, &mut z2);
    assert!(max_err(&z1, &z2) < 1e-14);
}

/// Timing for the 64×64×32 KNP grid. Run with
/// `cargo test --release -p ed-core --test linalg -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_64x64x32() {
    use std::time::Instant;
    let grid = Grid::new_3d([64, 64, 32], [15.0, 15.0, 15.0]);
    // Smoothly varying conductivity, as from KNP concentrations.
    let mut sigma = vec![0.0; grid.len()];
    let mut source = vec![0.0; grid.len()];
    grid.for_each_cell(|i, j, k, c| {
        let p = grid.cell_center(i, j, k);
        sigma[c] = 1.0 + 0.1 * (p[0] / 200.0).sin() * (p[1] / 300.0).cos();
        // A dipole-like pair of sources near the centre.
        let d1 = (i as f64 - 30.0).powi(2) + (j as f64 - 32.0).powi(2) + (k as f64 - 16.0).powi(2);
        let d2 = (i as f64 - 34.0).powi(2) + (j as f64 - 32.0).powi(2) + (k as f64 - 16.0).powi(2);
        source[c] = (-d1 / 4.0).exp() - (-d2 / 4.0).exp();
    });

    for (name, bc, ns) in [
        (
            "Neumann",
            BoundaryConditions::neumann(),
            NullSpace::Constant,
        ),
        (
            "Dirichlet",
            BoundaryConditions::dirichlet(),
            NullSpace::None,
        ),
    ] {
        let op = DiffusionOperator::new(grid, bc, &sigma);
        let opts = CgOptions {
            rtol: 1e-6,
            null_space: ns,
            ..Default::default()
        };
        let mut cg = Pcg::new(op.len());
        let mut mg = Multigrid::new(&op);
        let mut jac = Jacobi::new(&op);

        for (pname, m) in [
            ("Jacobi", &mut jac as &mut dyn Preconditioner),
            ("multigrid", &mut mg),
        ] {
            let mut x = vec![0.0; op.len()];
            let t = Instant::now();
            let cold = cg.solve(&op, m, &source, &mut x, &opts);
            let t_cold = t.elapsed();
            // Next "time step": sources change by 1 %, warm-start from the previous solution.
            let b2: Vec<f64> = source
                .iter()
                .enumerate()
                .map(|(c, s)| s * (1.0 + 0.01 * ((c % 7) as f64 / 7.0)))
                .collect();
            // Median of repeated warm solves, each from the same previous solution.
            let mut times = Vec::new();
            let mut warm = cold;
            let mut xw = x.clone();
            for _ in 0..15 {
                xw.copy_from_slice(&x);
                let t = Instant::now();
                warm = cg.solve(&op, m, &b2, &mut xw, &opts);
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "{name:9} {pname:9}: cold {:7.2} ms ({:3} it), warm median {:7.2} ms, best {:7.2} ms ({:3} it)",
                t_cold.as_secs_f64() * 1e3,
                cold.iterations,
                times[times.len() / 2],
                times[0],
                warm.iterations
            );
            assert!(cold.converged && warm.converged);
        }
    }
}
