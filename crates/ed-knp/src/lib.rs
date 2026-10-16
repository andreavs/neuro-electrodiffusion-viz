//! Tissue-scale KNP electrodiffusion with point-source neurons (Publication 1).
//!
//! The extracellular space is a porous medium on a uniform 3-D cell-centred
//! finite-volume grid. Each species obeys
//!
//! ```text
//! ∂c_k/∂t = ∇·(D̃_k ∇c_k) + ∇·((D̃_k z_k / ψ) c_k ∇φ) + f̃_k,   D̃_k = D_k / λ²
//! ```
//!
//! and the potential follows from electroneutrality: the ionic charge density
//! `ρ = F Σ_k z_k c_k` only changes where a neuron's capacitive current
//! charges its membrane, `∂ρ/∂t = −ĩ_cap`. With `σ = (F/ψ) Σ_k D̃_k z_k² c_k`
//! that gives one linear elliptic equation per step,
//!
//! ```text
//! −∇·(σ∇φ) = F Σ_k z_k ∇·(D̃_k ∇c_k) + F Σ_k z_k f̃_k + ĩ_cap.
//! ```
//!
//! Sign convention: point-source currents are positive out of the neuron into
//! the extracellular space, for both the ionic and the capacitive currents.
//!
//! # Time stepping
//!
//! Each step solves for φ with multigrid-preconditioned CG (warm-started from
//! the previous step), then advances the concentrations with forward Euler on
//! the same discrete fluxes. Explicit stepping is cheap here: the stability
//! limit `Δt ≤ 1 / (2 D̃_max Σ_a h_a⁻²)` is about 50 ms on the default grid,
//! far above the 0.1 ms step, and steps longer than the limit are split into
//! substeps automatically. Because the drift fluxes use exactly the face
//! conductances of the potential operator, total amounts are conserved to
//! round-off and the charge density changes only by `−ĩ_cap Δt`, up to the CG
//! tolerance. A semi-implicit scheme would need four extra solves per step
//! and would break that discrete consistency.

pub mod faces;

use ed_core::constants::T_DEFAULT;
use ed_core::ions::{IonSpecies, CA, CL, K, NA};

/// Generic anion X⁻ with the properties of Cl⁻ (Publication 1).
pub const X: IonSpecies = IonSpecies { name: "X-", ..CL };

/// An ion species with its baseline extracellular concentration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Species {
    pub ion: IonSpecies,
    /// Baseline concentration, mM. Used as the initial value and as the
    /// Dirichlet boundary value.
    pub baseline: f64,
}

/// Na⁺, K⁺, X⁻ and Ca²⁺ at the ECS baseline of Publication 1.
pub fn default_species() -> Vec<Species> {
    [(NA, 150.0), (K, 3.0), (X, 155.8), (CA, 1.4)]
        .into_iter()
        .map(|(ion, baseline)| Species { ion, baseline })
        .collect()
}

/// Condition on the outer boundary of the tissue box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterBoundary {
    /// Concentrations fixed at baseline and φ = 0 (as in the paper), on all six
    /// faces, including those of an axis with a single cell.
    Dirichlet,
    /// No flux of any species; φ is defined up to a constant and has zero mean.
    /// Charge cannot leave the box, so sources must carry zero net current
    /// (ionic plus capacitive); any excess is spread evenly over all cells.
    NoFlux,
}

/// Simulation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Cells per axis.
    pub cells: [usize; 3],
    /// Size of the box along each axis, µm.
    pub extent: [f64; 3],
    /// Time step, ms. Split into stable substeps when needed.
    pub dt: f64,
    /// Tortuosity λ: effective diffusion is `D / λ²`.
    pub tortuosity: f64,
    /// Volume fraction α: sources are scaled by `1 / α`.
    pub volume_fraction: f64,
    /// Temperature, K.
    pub temperature: f64,
    pub boundary: OuterBoundary,
    pub species: Vec<Species>,
    /// Relative CG tolerance of the potential solve.
    pub cg_rtol: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            cells: [64, 64, 32],
            extent: [1000.0, 1000.0, 500.0],
            dt: 0.1,
            tortuosity: 1.6,
            volume_fraction: 0.2,
            temperature: T_DEFAULT,
            boundary: OuterBoundary::Dirichlet,
            species: default_species(),
            cg_rtol: 1e-7,
        }
    }
}
