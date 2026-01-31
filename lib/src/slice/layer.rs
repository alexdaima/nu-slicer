//! Layer data structure.
//!
//! This module provides the Layer type representing a single horizontal slice
//! of a 3D model, mirroring BambuStudio's Layer class.

use crate::geometry::{ExPolygon, ExPolygons};
use crate::{Coord, CoordF};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a single layer of a sliced 3D model.
///
/// A layer contains the 2D geometry at a specific Z height, including:
/// - The slice contours (outlines of the model at this height)
/// - Layer regions (different printing regions with different settings)
/// - Support structures (if any)
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Layer {
    /// Layer index (0-based).
    id: usize,

    /// Z coordinate of the bottom of this layer (in scaled units).
    bottom_z: Coord,

    /// Z coordinate of the top of this layer (print_z, in scaled units).
    top_z: Coord,

    /// Height/thickness of this layer (in scaled units).
    height: Coord,

    /// The slice Z coordinate (middle of the layer, where slicing occurred).
    slice_z: Coord,

    /// The sliced regions at this layer.
    /// Each region may have different print settings (e.g., different extruder).
    regions: Vec<LayerRegion>,

    /// Reference to the layer below (if any).
    /// Used for calculating overhangs, bridges, etc.
    lower_layer_id: Option<usize>,

    /// Reference to the layer above (if any).
    upper_layer_id: Option<usize>,
}

impl Layer {
    /// Create a new layer.
    pub fn new(id: usize, bottom_z: Coord, top_z: Coord, slice_z: Coord) -> Self {
        Self {
            id,
            bottom_z,
            top_z,
            height: top_z - bottom_z,
            slice_z,
            regions: Vec::new(),
            lower_layer_id: None,
            upper_layer_id: None,
        }
    }

    /// Create a new layer with floating-point coordinates (in mm).
    pub fn new_f(id: usize, bottom_z: CoordF, top_z: CoordF, slice_z: CoordF) -> Self {
        use crate::scale;
        Self::new(id, scale(bottom_z), scale(top_z), scale(slice_z))
    }

    /// Get the layer ID.
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the bottom Z coordinate (scaled).
    #[inline]
    pub fn bottom_z(&self) -> Coord {
        self.bottom_z
    }

    /// Get the top Z coordinate / print Z (scaled).
    #[inline]
    pub fn top_z(&self) -> Coord {
        self.top_z
    }

    /// Get the print Z (alias for top_z).
    #[inline]
    pub fn print_z(&self) -> Coord {
        self.top_z
    }

    /// Get the layer height/thickness (scaled).
    #[inline]
    pub fn height(&self) -> Coord {
        self.height
    }

    /// Get the slice Z coordinate (scaled).
    #[inline]
    pub fn slice_z(&self) -> Coord {
        self.slice_z
    }

    /// Get the bottom Z coordinate in mm.
    #[inline]
    pub fn bottom_z_mm(&self) -> CoordF {
        crate::unscale(self.bottom_z)
    }

    /// Get the top Z coordinate in mm.
    #[inline]
    pub fn top_z_mm(&self) -> CoordF {
        crate::unscale(self.top_z)
    }

    /// Get the layer height in mm.
    #[inline]
    pub fn height_mm(&self) -> CoordF {
        crate::unscale(self.height)
    }

    /// Get the slice Z coordinate in mm.
    #[inline]
    pub fn slice_z_mm(&self) -> CoordF {
        crate::unscale(self.slice_z)
    }

    /// Get the layer regions.
    #[inline]
    pub fn regions(&self) -> &[LayerRegion] {
        &self.regions
    }

    /// Get mutable access to the layer regions.
    #[inline]
    pub fn regions_mut(&mut self) -> &mut Vec<LayerRegion> {
        &mut self.regions
    }

    /// Add a region to this layer.
    pub fn add_region(&mut self, region: LayerRegion) {
        self.regions.push(region);
    }

    /// Get the number of regions.
    #[inline]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Get a region by index.
    #[inline]
    pub fn region(&self, idx: usize) -> Option<&LayerRegion> {
        self.regions.get(idx)
    }

    /// Get a region by index (mutable).
    #[inline]
    pub fn region_mut(&mut self, idx: usize) -> Option<&mut LayerRegion> {
        self.regions.get_mut(idx)
    }

    /// Get or create a region at the given index.
    pub fn get_or_create_region(&mut self, idx: usize) -> &mut LayerRegion {
        while self.regions.len() <= idx {
            self.regions.push(LayerRegion::new());
        }
        &mut self.regions[idx]
    }

    /// Set the lower layer reference.
    #[inline]
    pub fn set_lower_layer(&mut self, id: Option<usize>) {
        self.lower_layer_id = id;
    }

    /// Set the upper layer reference.
    #[inline]
    pub fn set_upper_layer(&mut self, id: Option<usize>) {
        self.upper_layer_id = id;
    }

    /// Get the lower layer ID.
    #[inline]
    pub fn lower_layer_id(&self) -> Option<usize> {
        self.lower_layer_id
    }

    /// Get the upper layer ID.
    #[inline]
    pub fn upper_layer_id(&self) -> Option<usize> {
        self.upper_layer_id
    }

    /// Check if this is the first layer.
    #[inline]
    pub fn is_first_layer(&self) -> bool {
        self.id == 0
    }

    /// Check if this layer has a layer below it.
    #[inline]
    pub fn has_lower_layer(&self) -> bool {
        self.lower_layer_id.is_some()
    }

    /// Check if this layer has a layer above it.
    #[inline]
    pub fn has_upper_layer(&self) -> bool {
        self.upper_layer_id.is_some()
    }

    /// Get all slices (ExPolygons) from all regions.
    pub fn all_slices(&self) -> ExPolygons {
        let mut result = Vec::new();
        for region in &self.regions {
            result.extend(region.slices.iter().cloned());
        }
        result
    }

    /// Check if this layer is empty (no geometry).
    pub fn is_empty(&self) -> bool {
        self.regions.iter().all(|r| r.slices.is_empty())
    }

    /// Clear all geometry from this layer.
    pub fn clear(&mut self) {
        self.regions.clear();
    }
}

impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layer(id={}, z={:.3}mm, height={:.3}mm, {} regions)",
            self.id,
            self.top_z_mm(),
            self.height_mm(),
            self.regions.len()
        )
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layer {} at z={:.3}mm (height={:.3}mm)",
            self.id,
            self.top_z_mm(),
            self.height_mm()
        )
    }
}

/// Represents a region within a layer.
///
/// A layer region contains the geometry and toolpaths for a specific
/// region of the layer, which may have different print settings
/// (e.g., different extruder, infill pattern, etc.).
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct LayerRegion {
    /// The sliced contours for this region (ExPolygons with holes).
    pub slices: ExPolygons,

    /// Fill surfaces - internal classification of surfaces for infill.
    pub fill_surfaces: Vec<crate::slice::Surface>,

    /// Perimeter extrusion paths (to be generated).
    pub perimeters: Vec<crate::geometry::Polygon>,

    /// Thin fill paths (narrow areas that can't fit perimeters).
    pub thin_fills: Vec<crate::geometry::Polyline>,

    /// Fill extrusion paths (infill, to be generated).
    pub fills: Vec<crate::geometry::Polygon>,

    /// Region index (for multi-material/multi-region prints).
    pub region_id: usize,
}

impl LayerRegion {
    /// Create a new empty layer region.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a layer region with slices.
    pub fn with_slices(slices: ExPolygons) -> Self {
        Self {
            slices,
            ..Default::default()
        }
    }

    /// Create a layer region with a region ID.
    pub fn with_region_id(region_id: usize) -> Self {
        Self {
            region_id,
            ..Default::default()
        }
    }

    /// Check if this region is empty.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Get the total area of all slices.
    pub fn area(&self) -> CoordF {
        self.slices.iter().map(|s| s.area()).sum()
    }

    /// Clear all geometry.
    pub fn clear(&mut self) {
        self.slices.clear();
        self.fill_surfaces.clear();
        self.perimeters.clear();
        self.thin_fills.clear();
        self.fills.clear();
    }

    /// Add a slice to this region.
    pub fn add_slice(&mut self, slice: ExPolygon) {
        self.slices.push(slice);
    }

    /// Add multiple slices to this region.
    pub fn add_slices(&mut self, slices: impl IntoIterator<Item = ExPolygon>) {
        self.slices.extend(slices);
    }

    /// Set the slices for this region.
    pub fn set_slices(&mut self, slices: ExPolygons) {
        self.slices = slices;
    }

    /// Get the slices for this region.
    pub fn slices(&self) -> &ExPolygons {
        &self.slices
    }

    /// Get mutable access to the slices.
    pub fn slices_mut(&mut self) -> &mut ExPolygons {
        &mut self.slices
    }
}

impl fmt::Debug for LayerRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LayerRegion(region_id={}, {} slices, {} surfaces)",
            self.region_id,
            self.slices.len(),
            self.fill_surfaces.len()
        )
    }
}

/// Type alias for a collection of layers.
pub type Layers = Vec<Layer>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Point, Polygon};

    #[test]
    fn test_layer_new() {
        let layer = Layer::new_f(0, 0.0, 0.2, 0.1);
        assert_eq!(layer.id(), 0);
        assert!((layer.height_mm() - 0.2).abs() < 1e-6);
        assert!((layer.slice_z_mm() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_layer_regions() {
        let mut layer = Layer::new_f(0, 0.0, 0.2, 0.1);
        assert_eq!(layer.region_count(), 0);
        assert!(layer.is_empty());

        layer.add_region(LayerRegion::new());
        assert_eq!(layer.region_count(), 1);

        layer.add_region(LayerRegion::with_region_id(1));
        assert_eq!(layer.region_count(), 2);
        assert_eq!(layer.region(1).unwrap().region_id, 1);
    }

    #[test]
    fn test_layer_get_or_create_region() {
        let mut layer = Layer::new_f(0, 0.0, 0.2, 0.1);
        assert_eq!(layer.region_count(), 0);

        let _region = layer.get_or_create_region(2);
        assert_eq!(layer.region_count(), 3);
    }

    #[test]
    fn test_layer_is_first() {
        let layer0 = Layer::new_f(0, 0.0, 0.2, 0.1);
        let layer1 = Layer::new_f(1, 0.2, 0.4, 0.3);

        assert!(layer0.is_first_layer());
        assert!(!layer1.is_first_layer());
    }

    #[test]
    fn test_layer_links() {
        let mut layer = Layer::new_f(1, 0.2, 0.4, 0.3);

        assert!(!layer.has_lower_layer());
        assert!(!layer.has_upper_layer());

        layer.set_lower_layer(Some(0));
        layer.set_upper_layer(Some(2));

        assert!(layer.has_lower_layer());
        assert!(layer.has_upper_layer());
        assert_eq!(layer.lower_layer_id(), Some(0));
        assert_eq!(layer.upper_layer_id(), Some(2));
    }

    #[test]
    fn test_layer_region_area() {
        let mut region = LayerRegion::new();
        assert!(region.is_empty());

        // Add a simple square
        let square = Polygon::rectangle(Point::new(0, 0), Point::new(1000000, 1000000));
        region.add_slice(square.into());

        assert!(!region.is_empty());
        // Area should be 1mm² = 1e12 in scaled units
        let area = region.area();
        assert!(area > 0.0);
    }

    #[test]
    fn test_layer_all_slices() {
        let mut layer = Layer::new_f(0, 0.0, 0.2, 0.1);

        let square = Polygon::rectangle(Point::new(0, 0), Point::new(100, 100));

        let mut region1 = LayerRegion::new();
        region1.add_slice(square.clone().into());
        layer.add_region(region1);

        let mut region2 = LayerRegion::new();
        region2.add_slice(square.into());
        layer.add_region(region2);

        let all_slices = layer.all_slices();
        assert_eq!(all_slices.len(), 2);
    }

    #[test]
    fn test_layer_clear() {
        let mut layer = Layer::new_f(0, 0.0, 0.2, 0.1);
        layer.add_region(LayerRegion::new());
        layer.add_region(LayerRegion::new());

        assert_eq!(layer.region_count(), 2);

        layer.clear();
        assert_eq!(layer.region_count(), 0);
    }
}
