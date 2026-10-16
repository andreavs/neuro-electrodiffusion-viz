//! Shared physics, units and numerics for the three simulation scales.
//!
//! Unit conventions (used by every crate in the workspace):
//! - length: µm
//! - time: ms
//! - concentration: mM (= mol/m³)
//! - potential: mV
//! - diffusion coefficients: µm²/ms
//!
//! Scale-specific solvers live in their own crates and implement [`Simulation`].

pub mod constants;
pub mod ions;
pub mod linalg;
pub mod membrane;

/// Common interface implemented by every scale's simulator so the WASM layer
/// and the browser worker can drive them uniformly.
pub trait Simulation {
    /// Advance the simulation by `n` internal time steps.
    fn step(&mut self, n: usize);
    /// Current simulated time in ms.
    fn time_ms(&self) -> f64;
}
