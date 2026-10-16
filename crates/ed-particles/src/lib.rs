//! Molecular-scale Brownian electrodiffusion of individual ions (Publication 3).
//!
//! Each particle moves r ← r + X + (z D Δt / ψ) E(r), X ~ N(0, 2DΔt) per axis, inside a
//! dendritic spine whose walls reflect specularly and whose neck base absorbs. The
//! quantity of interest is the mean first passage time (MFPT) from the head centre.
//!
//! ```
//! use ed_core::Simulation;
//! use ed_particles::{NeckField, ParticleParams, ParticleSim, SpineGeometry};
//!
//! let spine = SpineGeometry::sharp();
//! let mut sim = ParticleSim::new(spine, ParticleParams::default(), 42)
//!     .with_field(NeckField::for_spine(&spine, 10.0));
//! sim.release(1000);
//! sim.step(100);
//! assert_eq!(sim.positions_xyz().len(), 3 * sim.n_alive());
//! ```

pub mod field;
pub mod geometry;
pub mod rng;
pub mod sim;

pub use field::{ElectricField, NeckField, NoField, UniformField};
pub use geometry::{FreeSpace, Geometry, Junction, SegmentExit, SpineGeometry, Vec3};
pub use rng::Xoshiro256pp;
pub use sim::{
    resolve_step, MfptEstimate, ParticleParams, ParticleSim, StepOutcome, CA_SPINE, DT_DEFAULT,
    MAX_PARTICLES,
};
