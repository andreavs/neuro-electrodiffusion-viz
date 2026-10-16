//! Cell-scale KNP-EMI electrodiffusion with explicit membranes (Publication 2).
//!
//! Space is discretised with P1 finite elements on conforming tetrahedral meshes, one per
//! subdomain, with matching facets on the membrane ([`mesh`]). See docs/ARCHITECTURE.md for
//! the model and the time-stepping scheme.

pub mod geometry;
pub mod mesh;

pub use geometry::{AxonBundle, Geometry, MembraneFace};
