//! Print module - orchestrates the printing process.
//!
//! This module provides high-level types for managing print jobs:
//! - [`Print`] - Represents an entire print job
//! - [`PrintObject`] - Represents a single object to be printed

use crate::geometry::BoundingBox3F;
use crate::mesh::TriangleMesh;
use crate::slice::Layer;

/// Represents an entire print job containing one or more objects.
#[derive(Debug, Default)]
pub struct Print {
    /// Objects to be printed
    objects: Vec<PrintObject>,
}

impl Print {
    /// Create a new empty print job.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Add an object to the print job.
    pub fn add_object(&mut self, object: PrintObject) {
        self.objects.push(object);
    }

    /// Get the objects in this print job.
    pub fn objects(&self) -> &[PrintObject] {
        &self.objects
    }

    /// Get mutable access to the objects.
    pub fn objects_mut(&mut self) -> &mut [PrintObject] {
        &mut self.objects
    }

    /// Get the number of objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Check if the print job is empty.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

/// Represents a single object to be printed.
#[derive(Debug)]
pub struct PrintObject {
    /// The mesh for this object
    mesh: TriangleMesh,
    /// Sliced layers (populated after slicing)
    layers: Vec<Layer>,
    /// Object name/identifier
    name: String,
}

impl PrintObject {
    /// Create a new print object from a mesh.
    pub fn new(mesh: TriangleMesh) -> Self {
        Self {
            mesh,
            layers: Vec::new(),
            name: String::new(),
        }
    }

    /// Create a new print object with a name.
    pub fn with_name(mesh: TriangleMesh, name: impl Into<String>) -> Self {
        Self {
            mesh,
            layers: Vec::new(),
            name: name.into(),
        }
    }

    /// Get the mesh.
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Get the sliced layers.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Set the sliced layers.
    pub fn set_layers(&mut self, layers: Vec<Layer>) {
        self.layers = layers;
    }

    /// Get the object name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the object name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Get the bounding box of the mesh.
    pub fn bounding_box(&self) -> BoundingBox3F {
        self.mesh.compute_bounding_box()
    }

    /// Check if this object has been sliced.
    pub fn is_sliced(&self) -> bool {
        !self.layers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_new() {
        let print = Print::new();
        assert!(print.is_empty());
        assert_eq!(print.object_count(), 0);
    }

    #[test]
    fn test_print_object_new() {
        let mesh = TriangleMesh::new();
        let obj = PrintObject::new(mesh);
        assert!(!obj.is_sliced());
        assert!(obj.name().is_empty());
    }

    #[test]
    fn test_print_object_with_name() {
        let mesh = TriangleMesh::new();
        let obj = PrintObject::with_name(mesh, "test_object");
        assert_eq!(obj.name(), "test_object");
    }
}
