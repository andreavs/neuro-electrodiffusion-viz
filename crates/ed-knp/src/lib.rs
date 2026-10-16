//! Tissue-scale KNP electrodiffusion with point-source neurons (Publication 1).
//!
//! [`KnpSolver`] advances ion concentrations in the extracellular space (ECS) of a porous
//! medium on a uniform cell-centred 3-D grid, with the potential from the electroneutral
//! Kirchhoff-Nernst-Planck equation (Solbrå et al. 2018, Eq 12):
//!
//! ```text
//! ∂c_k/∂t = -∇·J_k + f̃_k,     J_k = -D̃_k ∇c_k - (D̃_k z_k / ψ) c_k ∇φ
//! -∇·(σ∇φ) = Σ_k z_k ∇·(D̃_k ∇c_k) + Σ_k z_k f̃_k + ĩ_cap / F,     σ = Σ_k D̃_k z_k² c_k / ψ
//! ```
//!
//! with `D̃ = D/λ²` and sources divided by the volume fraction `α`. `σ` here is the
//! conductivity divided by `F`, so every term is in mM/ms. Membrane currents are positive
//! when positive charge leaves the neuron, so the source term is the total membrane
//! current, as in volume-conductor theory (Eq 40), and the ECS charge changes only by
//! `∂ρ/∂t = -ĩ_cap`: the charge stored on the outside of the membrane.
//!
//! See [`KnpSolver::step`] for the time stepping.

mod flux;
mod solver;

pub use solver::{KnpSolver, StepStats, CHARGE_TOL};

use ed_core::ions::{IonSpecies, CA, CL, K, NA};

/// The generic anion X⁻, which has the properties of Cl⁻.
pub const X: IonSpecies = IonSpecies { name: "X-", ..CL };

/// An ion species with its baseline ECS concentration (mM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ion {
    pub species: IonSpecies,
    pub baseline: f64,
}

/// Outer boundary condition, applied on every face of the box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterBoundary {
    /// Concentrations fixed at their baselines and `φ = 0`, as in the paper.
    Dirichlet,
    /// No flux of any species; `φ` is fixed by giving it zero mean.
    NoFlux,
}

/// Parameters of a [`KnpSolver`].
#[derive(Clone, Debug, PartialEq)]
pub struct KnpParams {
    /// Cells per axis. Even counts (such as 64×64×32) suit the multigrid preconditioner.
    pub cells: [usize; 3],
    /// Size of the box (µm), with its lower corner at the origin.
    pub extent: [f64; 3],
    /// Tortuosity λ: `D̃ = D/λ²`.
    pub tortuosity: f64,
    /// ECS volume fraction α: sources are divided by it.
    pub volume_fraction: f64,
    pub ions: Vec<Ion>,
    pub boundary: OuterBoundary,
    /// Time step (ms). [`KnpSolver::step`] subdivides it if it exceeds the stability limit.
    pub dt: f64,
    /// Temperature (K).
    pub temperature: f64,
}

impl Default for KnpParams {
    /// 64×64×32 cells over 1000×1000×500 µm with the Publication 1 ions (Table 2).
    fn default() -> Self {
        let ion = |species, baseline| Ion { species, baseline };
        Self {
            cells: [64, 64, 32],
            extent: [1000.0, 1000.0, 500.0],
            tortuosity: 1.6,
            volume_fraction: 0.2,
            ions: vec![ion(NA, 150.0), ion(K, 3.0), ion(X, 155.8), ion(CA, 1.4)],
            boundary: OuterBoundary::Dirichlet,
            dt: 0.1,
            temperature: ed_core::constants::T_DEFAULT,
        }
    }
}

/// Converts a current in nA into a charge rate in mM·µm³/ms (`10⁶ / F`).
pub const NA_TO_MM_UM3_PER_MS: f64 = 1e6 / ed_core::constants::F;
