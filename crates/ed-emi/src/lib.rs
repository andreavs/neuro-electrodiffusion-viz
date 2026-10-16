//! Cell-scale KNP-EMI electrodiffusion with explicit membranes (Publication 2).
//!
//! Space is discretised with P1 finite elements on conforming tetrahedral meshes, one per
//! subdomain, with matching facets on the membrane ([`mesh`], [`fem`]). See docs/ARCHITECTURE.md for
//! the model and the time-stepping scheme.

pub mod fem;
pub mod mesh;
pub mod solver;
