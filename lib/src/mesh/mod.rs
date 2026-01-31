//! Mesh loading and processing.
//!
//! This module provides types and functions for working with triangle meshes:
//! - [`TriangleMesh`] - The main triangle mesh data structure
//! - [`Triangle`] - A single triangle
//! - STL file loading and saving
//! - Mesh repair and validation

mod stl;
mod triangle_mesh;

pub use stl::{load_stl, save_stl};
pub use triangle_mesh::{Triangle, TriangleMesh};
