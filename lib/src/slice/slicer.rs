//! Slicer - Core slicing engine.
//!
//! This module provides the main slicing functionality that converts
//! a 3D mesh into a series of 2D layers, mirroring BambuStudio's
//! TriangleMeshSlicer.

use crate::geometry::ExPolygons;
use crate::mesh::TriangleMesh;
use crate::slice::mesh_slicer;
use crate::slice::{Layer, LayerRegion, SlicingParams};
use crate::{CoordF, Error, Result};
use std::fmt;

/// The main slicer engine that converts meshes into layers.
///
/// This is the core of the slicing pipeline. Given a mesh and slicing
/// parameters, it produces a series of layers with 2D contours.
pub struct Slicer {
    /// Slicing parameters.
    params: SlicingParams,
}

impl Slicer {
    /// Create a new slicer with the given parameters.
    pub fn new(params: SlicingParams) -> Self {
        Self { params }
    }

    /// Create a new slicer with default parameters.
    pub fn with_defaults() -> Self {
        Self::new(SlicingParams::default())
    }

    /// Get the slicing parameters.
    pub fn params(&self) -> &SlicingParams {
        &self.params
    }

    /// Get mutable access to the slicing parameters.
    pub fn params_mut(&mut self) -> &mut SlicingParams {
        &mut self.params
    }

    /// Slice a mesh into layers.
    ///
    /// This is the main entry point for slicing. It computes the Z heights
    /// for each layer and slices the mesh at those heights.
    pub fn slice(&self, mesh: &TriangleMesh) -> Result<Vec<Layer>> {
        self.slice_with_callback(mesh, |_| {})
    }

    /// Slice a mesh into layers with a progress callback.
    ///
    /// The callback receives a progress value from 0.0 to 1.0.
    pub fn slice_with_callback<F>(&self, mesh: &TriangleMesh, mut callback: F) -> Result<Vec<Layer>>
    where
        F: FnMut(f64),
    {
        if mesh.is_empty() {
            return Err(Error::Mesh("Cannot slice an empty mesh".into()));
        }

        // Calculate layer heights
        let z_heights = self.compute_layer_heights(mesh)?;
        if z_heights.is_empty() {
            return Err(Error::Slicing("No layers to slice".into()));
        }

        callback(0.1);

        // Extract slice Z values for the mesh slicer
        let slice_zs: Vec<CoordF> = z_heights.iter().map(|h| h.slice_z).collect();

        // Perform actual mesh slicing
        let sliced_expolygons = mesh_slicer::slice_mesh(mesh, &slice_zs);

        callback(0.6);

        // Build layers from sliced geometry
        let layers = self.build_layers(&z_heights, sliced_expolygons, |progress| {
            callback(0.6 + progress * 0.4);
        })?;

        callback(1.0);
        Ok(layers)
    }

    /// Compute the Z heights for each layer based on slicing parameters.
    fn compute_layer_heights(&self, mesh: &TriangleMesh) -> Result<Vec<LayerHeight>> {
        let bb = mesh.compute_bounding_box();
        if !bb.is_defined() {
            return Err(Error::Mesh("Mesh has no bounding box".into()));
        }

        let min_z = bb.min.z;
        let max_z = bb.max.z;
        let object_height = max_z - min_z;

        if object_height <= 0.0 {
            return Err(Error::Mesh("Object has zero height".into()));
        }

        let first_layer_height = self.params.first_layer_height;
        let layer_height = self.params.layer_height;

        let mut heights = Vec::new();
        let mut z = min_z;

        // First layer
        if z < max_z {
            let top_z = (z + first_layer_height).min(max_z);
            let slice_z = (z + top_z) / 2.0;
            heights.push(LayerHeight {
                bottom_z: z,
                top_z,
                slice_z,
            });
            z = top_z;
        }

        // Subsequent layers
        while z < max_z {
            let top_z = (z + layer_height).min(max_z);
            let slice_z = (z + top_z) / 2.0;
            heights.push(LayerHeight {
                bottom_z: z,
                top_z,
                slice_z,
            });
            z = top_z;
        }

        Ok(heights)
    }

    /// Build layers from sliced geometry.
    fn build_layers<F>(
        &self,
        heights: &[LayerHeight],
        sliced_geometry: Vec<ExPolygons>,
        mut callback: F,
    ) -> Result<Vec<Layer>>
    where
        F: FnMut(f64),
    {
        let total = heights.len();
        let mut layers: Vec<Layer> = Vec::with_capacity(total);

        for (i, (h, expolygons)) in heights.iter().zip(sliced_geometry.into_iter()).enumerate() {
            let mut layer = Layer::new_f(i, h.bottom_z, h.top_z, h.slice_z);

            // Create a layer region with the sliced geometry
            let mut region = LayerRegion::new();
            region.set_slices(expolygons);
            layer.add_region(region);

            // Set layer links
            if i > 0 {
                layer.set_lower_layer(Some(i - 1));
            }
            if i < total - 1 {
                layer.set_upper_layer(Some(i + 1));
            }

            layers.push(layer);

            if i % 10 == 0 {
                callback(i as f64 / total as f64);
            }
        }

        callback(1.0);
        Ok(layers)
    }

    /// Slice the mesh at a single Z height, returning ExPolygons.
    ///
    /// This is the core slicing function that computes the intersection
    /// of the mesh with a horizontal plane at the given Z coordinate.
    pub fn slice_at_z(&self, mesh: &TriangleMesh, z: CoordF) -> Result<ExPolygons> {
        if mesh.is_empty() {
            return Err(Error::Mesh("Cannot slice an empty mesh".into()));
        }
        Ok(mesh_slicer::slice_mesh_at_z(mesh, z))
    }
}

impl Default for Slicer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl fmt::Debug for Slicer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Slicer({:?})", self.params)
    }
}

/// Internal struct to hold layer height information.
#[derive(Clone, Copy, Debug)]
struct LayerHeight {
    /// Bottom Z coordinate.
    bottom_z: CoordF,
    /// Top Z coordinate (print Z).
    top_z: CoordF,
    /// Slice Z coordinate (where the slicing plane is).
    slice_z: CoordF,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriangleMesh;

    #[test]
    fn test_slicer_new() {
        let params = SlicingParams::default();
        let slicer = Slicer::new(params);
        assert!((slicer.params().layer_height - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_compute_layer_heights() {
        let slicer = Slicer::with_defaults();
        let mesh = TriangleMesh::cube(10.0); // 10mm cube

        let heights = slicer.compute_layer_heights(&mesh).unwrap();

        // Should have multiple layers
        assert!(!heights.is_empty());

        // First layer should start at the bottom
        assert!((heights[0].bottom_z - (-5.0)).abs() < 1e-6);

        // Layers should be contiguous
        for i in 1..heights.len() {
            assert!((heights[i].bottom_z - heights[i - 1].top_z).abs() < 1e-6);
        }
    }

    #[test]
    fn test_slice_cube() {
        let slicer = Slicer::with_defaults();
        let mesh = TriangleMesh::cube(10.0);

        let layers = slicer.slice(&mesh).unwrap();

        // Should have multiple layers
        assert!(!layers.is_empty());

        // Check layer IDs are sequential
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(layer.id(), i);
        }

        // Check layer links
        for i in 0..layers.len() {
            if i > 0 {
                assert_eq!(layers[i].lower_layer_id(), Some(i - 1));
            } else {
                assert_eq!(layers[i].lower_layer_id(), None);
            }
            if i < layers.len() - 1 {
                assert_eq!(layers[i].upper_layer_id(), Some(i + 1));
            } else {
                assert_eq!(layers[i].upper_layer_id(), None);
            }
        }

        // Check that layers have actual geometry from slicing
        for layer in &layers {
            // Each layer should have at least one region
            assert!(!layer.regions().is_empty(), "Layer should have regions");
        }
    }

    #[test]
    fn test_slice_cube_has_geometry() {
        let slicer = Slicer::with_defaults();
        let mesh = TriangleMesh::cube(10.0);

        let layers = slicer.slice(&mesh).unwrap();

        // Count layers with actual geometry
        let layers_with_geometry = layers
            .iter()
            .filter(|l| l.regions().iter().any(|r| !r.slices().is_empty()))
            .count();

        // Most layers should have geometry (the cube spans all layers)
        assert!(
            layers_with_geometry > layers.len() / 2,
            "Expected most layers to have geometry, got {}/{}",
            layers_with_geometry,
            layers.len()
        );
    }

    #[test]
    fn test_slice_empty_mesh() {
        let slicer = Slicer::with_defaults();
        let mesh = TriangleMesh::new();

        let result = slicer.slice(&mesh);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_with_callback() {
        let slicer = Slicer::with_defaults();
        let mesh = TriangleMesh::cube(10.0);

        let mut last_progress = 0.0;
        let layers = slicer
            .slice_with_callback(&mesh, |progress| {
                assert!(progress >= last_progress);
                last_progress = progress;
            })
            .unwrap();

        assert!(!layers.is_empty());
        assert!((last_progress - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_slice_at_z() {
        let slicer = Slicer::with_defaults();
        let mesh = TriangleMesh::cube(10.0);

        // Slice at the middle of the cube
        let expolygons = slicer.slice_at_z(&mesh, 0.0).unwrap();

        // Should have exactly one contour (the cube cross-section)
        assert_eq!(
            expolygons.len(),
            1,
            "Expected 1 contour for cube slice at z=0"
        );

        // The contour should be a square with no holes
        assert!(
            expolygons[0].holes.is_empty(),
            "Cube cross-section should have no holes"
        );
    }
}
