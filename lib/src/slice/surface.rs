//! Surface types for layer regions.
//!
//! This module provides the Surface type representing classified regions
//! within a layer, mirroring BambuStudio's Surface class.

use crate::geometry::ExPolygon;
use crate::CoordF;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Classification of a surface within a layer.
///
/// Surfaces are classified to determine how they should be filled:
/// - Top/bottom surfaces get solid infill
/// - Internal surfaces get sparse infill
/// - Bridge surfaces need special handling
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SurfaceType {
    /// Top surface (visible from above).
    Top,
    /// Bottom surface (visible from below, or first layer).
    Bottom,
    /// Bottom surface that bridges over air/support.
    BottomBridge,
    /// Internal solid surface (between top/bottom and infill).
    #[default]
    InternalSolid,
    /// Internal surface that will receive sparse infill.
    Internal,
    /// Internal bridge surface.
    InternalBridge,
    /// Internal void (empty space, no infill).
    InternalVoid,
}

impl SurfaceType {
    /// Check if this surface type is a top surface.
    #[inline]
    pub fn is_top(&self) -> bool {
        matches!(self, SurfaceType::Top)
    }

    /// Check if this surface type is a bottom surface.
    #[inline]
    pub fn is_bottom(&self) -> bool {
        matches!(self, SurfaceType::Bottom | SurfaceType::BottomBridge)
    }

    /// Check if this surface type is a bridge.
    #[inline]
    pub fn is_bridge(&self) -> bool {
        matches!(
            self,
            SurfaceType::BottomBridge | SurfaceType::InternalBridge
        )
    }

    /// Check if this surface type requires solid infill.
    #[inline]
    pub fn is_solid(&self) -> bool {
        matches!(
            self,
            SurfaceType::Top
                | SurfaceType::Bottom
                | SurfaceType::BottomBridge
                | SurfaceType::InternalSolid
                | SurfaceType::InternalBridge
        )
    }

    /// Check if this surface type is internal (not top or bottom).
    #[inline]
    pub fn is_internal(&self) -> bool {
        matches!(
            self,
            SurfaceType::Internal
                | SurfaceType::InternalSolid
                | SurfaceType::InternalBridge
                | SurfaceType::InternalVoid
        )
    }

    /// Check if this surface type is external (top or bottom).
    #[inline]
    pub fn is_external(&self) -> bool {
        !self.is_internal()
    }

    /// Get a human-readable name for this surface type.
    pub fn name(&self) -> &'static str {
        match self {
            SurfaceType::Top => "top",
            SurfaceType::Bottom => "bottom",
            SurfaceType::BottomBridge => "bottom bridge",
            SurfaceType::InternalSolid => "internal solid",
            SurfaceType::Internal => "internal",
            SurfaceType::InternalBridge => "internal bridge",
            SurfaceType::InternalVoid => "internal void",
        }
    }
}

impl fmt::Display for SurfaceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A surface is a classified region within a layer.
///
/// Each surface has a type (determining how it should be filled)
/// and geometry (the ExPolygon defining its shape).
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Surface {
    /// The geometry of this surface.
    pub expolygon: ExPolygon,

    /// The type/classification of this surface.
    pub surface_type: SurfaceType,

    /// Thickness of this surface (layer height), in mm.
    pub thickness: CoordF,

    /// Thickness of the layer below, in mm (for bridge calculations).
    pub thickness_layers: usize,

    /// Bridge angle in radians (for bridge surfaces).
    /// None if not a bridge or angle not yet determined.
    pub bridge_angle: Option<CoordF>,

    /// Extra perimeters needed for this surface.
    pub extra_perimeters: usize,
}

impl Surface {
    /// Create a new surface with the given geometry and type.
    pub fn new(expolygon: ExPolygon, surface_type: SurfaceType) -> Self {
        Self {
            expolygon,
            surface_type,
            thickness: 0.0,
            thickness_layers: 1,
            bridge_angle: None,
            extra_perimeters: 0,
        }
    }

    /// Create a new top surface.
    pub fn top(expolygon: ExPolygon) -> Self {
        Self::new(expolygon, SurfaceType::Top)
    }

    /// Create a new bottom surface.
    pub fn bottom(expolygon: ExPolygon) -> Self {
        Self::new(expolygon, SurfaceType::Bottom)
    }

    /// Create a new internal surface.
    pub fn internal(expolygon: ExPolygon) -> Self {
        Self::new(expolygon, SurfaceType::Internal)
    }

    /// Create a new internal solid surface.
    pub fn internal_solid(expolygon: ExPolygon) -> Self {
        Self::new(expolygon, SurfaceType::InternalSolid)
    }

    /// Create a new bridge surface.
    pub fn bridge(expolygon: ExPolygon, angle: Option<CoordF>) -> Self {
        Self {
            expolygon,
            surface_type: SurfaceType::BottomBridge,
            thickness: 0.0,
            thickness_layers: 1,
            bridge_angle: angle,
            extra_perimeters: 0,
        }
    }

    /// Check if this surface is empty (no geometry).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.expolygon.is_empty()
    }

    /// Get the area of this surface.
    #[inline]
    pub fn area(&self) -> CoordF {
        self.expolygon.area()
    }

    /// Check if this is a top surface.
    #[inline]
    pub fn is_top(&self) -> bool {
        self.surface_type.is_top()
    }

    /// Check if this is a bottom surface.
    #[inline]
    pub fn is_bottom(&self) -> bool {
        self.surface_type.is_bottom()
    }

    /// Check if this is a bridge surface.
    #[inline]
    pub fn is_bridge(&self) -> bool {
        self.surface_type.is_bridge()
    }

    /// Check if this is a solid surface.
    #[inline]
    pub fn is_solid(&self) -> bool {
        self.surface_type.is_solid()
    }

    /// Check if this is an internal surface.
    #[inline]
    pub fn is_internal(&self) -> bool {
        self.surface_type.is_internal()
    }

    /// Check if this is an external surface.
    #[inline]
    pub fn is_external(&self) -> bool {
        self.surface_type.is_external()
    }

    /// Set the surface type.
    pub fn set_type(&mut self, surface_type: SurfaceType) {
        self.surface_type = surface_type;
    }

    /// Set the bridge angle.
    pub fn set_bridge_angle(&mut self, angle: CoordF) {
        self.bridge_angle = Some(angle);
    }

    /// Set the thickness.
    pub fn set_thickness(&mut self, thickness: CoordF) {
        self.thickness = thickness;
    }

    /// Set the number of thickness layers.
    pub fn set_thickness_layers(&mut self, layers: usize) {
        self.thickness_layers = layers;
    }
}

impl fmt::Debug for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Surface({:?}, area={:.2}mm²)",
            self.surface_type,
            self.area()
        )
    }
}

impl fmt::Display for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} surface (area={:.2}mm²)",
            self.surface_type,
            self.area()
        )
    }
}

impl From<ExPolygon> for Surface {
    fn from(expolygon: ExPolygon) -> Self {
        Self::new(expolygon, SurfaceType::default())
    }
}

/// Type alias for a collection of surfaces.
pub type Surfaces = Vec<Surface>;

/// Collection of surfaces with utility methods.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SurfaceCollection {
    /// The surfaces in this collection.
    pub surfaces: Vec<Surface>,
}

impl SurfaceCollection {
    /// Create a new empty surface collection.
    pub fn new() -> Self {
        Self {
            surfaces: Vec::new(),
        }
    }

    /// Create a surface collection from a vector of surfaces.
    pub fn from_surfaces(surfaces: Vec<Surface>) -> Self {
        Self { surfaces }
    }

    /// Check if the collection is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.surfaces.is_empty()
    }

    /// Get the number of surfaces.
    #[inline]
    pub fn len(&self) -> usize {
        self.surfaces.len()
    }

    /// Add a surface to the collection.
    pub fn push(&mut self, surface: Surface) {
        self.surfaces.push(surface);
    }

    /// Clear all surfaces.
    pub fn clear(&mut self) {
        self.surfaces.clear();
    }

    /// Get all surfaces of a specific type.
    pub fn filter_by_type(&self, surface_type: SurfaceType) -> Vec<&Surface> {
        self.surfaces
            .iter()
            .filter(|s| s.surface_type == surface_type)
            .collect()
    }

    /// Get all top surfaces.
    pub fn top_surfaces(&self) -> Vec<&Surface> {
        self.surfaces.iter().filter(|s| s.is_top()).collect()
    }

    /// Get all bottom surfaces.
    pub fn bottom_surfaces(&self) -> Vec<&Surface> {
        self.surfaces.iter().filter(|s| s.is_bottom()).collect()
    }

    /// Get all solid surfaces.
    pub fn solid_surfaces(&self) -> Vec<&Surface> {
        self.surfaces.iter().filter(|s| s.is_solid()).collect()
    }

    /// Get all bridge surfaces.
    pub fn bridge_surfaces(&self) -> Vec<&Surface> {
        self.surfaces.iter().filter(|s| s.is_bridge()).collect()
    }

    /// Get the total area of all surfaces.
    pub fn total_area(&self) -> CoordF {
        self.surfaces.iter().map(|s| s.area()).sum()
    }

    /// Check if any surface has the given type.
    pub fn has_type(&self, surface_type: SurfaceType) -> bool {
        self.surfaces.iter().any(|s| s.surface_type == surface_type)
    }
}

impl fmt::Debug for SurfaceCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SurfaceCollection({} surfaces)", self.surfaces.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Point, Polygon};

    fn make_square_expolygon() -> ExPolygon {
        let poly = Polygon::rectangle(Point::new(0, 0), Point::new(1000000, 1000000));
        ExPolygon::new(poly)
    }

    #[test]
    fn test_surface_type_classification() {
        assert!(SurfaceType::Top.is_top());
        assert!(!SurfaceType::Top.is_bottom());
        assert!(SurfaceType::Top.is_solid());
        assert!(SurfaceType::Top.is_external());

        assert!(SurfaceType::Bottom.is_bottom());
        assert!(SurfaceType::Bottom.is_solid());
        assert!(SurfaceType::Bottom.is_external());

        assert!(SurfaceType::BottomBridge.is_bottom());
        assert!(SurfaceType::BottomBridge.is_bridge());
        assert!(SurfaceType::BottomBridge.is_solid());

        assert!(SurfaceType::Internal.is_internal());
        assert!(!SurfaceType::Internal.is_solid());

        assert!(SurfaceType::InternalSolid.is_internal());
        assert!(SurfaceType::InternalSolid.is_solid());

        assert!(SurfaceType::InternalBridge.is_bridge());
        assert!(SurfaceType::InternalBridge.is_solid());
    }

    #[test]
    fn test_surface_new() {
        let expoly = make_square_expolygon();
        let surface = Surface::new(expoly, SurfaceType::Top);

        assert!(surface.is_top());
        assert!(surface.is_solid());
        assert!(!surface.is_empty());
        assert!(surface.area() > 0.0);
    }

    #[test]
    fn test_surface_constructors() {
        let expoly = make_square_expolygon();

        let top = Surface::top(expoly.clone());
        assert!(top.is_top());

        let bottom = Surface::bottom(expoly.clone());
        assert!(bottom.is_bottom());

        let internal = Surface::internal(expoly.clone());
        assert!(internal.is_internal());
        assert!(!internal.is_solid());

        let bridge = Surface::bridge(expoly.clone(), Some(0.5));
        assert!(bridge.is_bridge());
        assert_eq!(bridge.bridge_angle, Some(0.5));
    }

    #[test]
    fn test_surface_setters() {
        let expoly = make_square_expolygon();
        let mut surface = Surface::new(expoly, SurfaceType::Internal);

        surface.set_type(SurfaceType::Top);
        assert!(surface.is_top());

        surface.set_bridge_angle(1.5);
        assert_eq!(surface.bridge_angle, Some(1.5));

        surface.set_thickness(0.2);
        assert!((surface.thickness - 0.2).abs() < 1e-6);

        surface.set_thickness_layers(3);
        assert_eq!(surface.thickness_layers, 3);
    }

    #[test]
    fn test_surface_collection() {
        let expoly = make_square_expolygon();

        let mut collection = SurfaceCollection::new();
        assert!(collection.is_empty());

        collection.push(Surface::top(expoly.clone()));
        collection.push(Surface::bottom(expoly.clone()));
        collection.push(Surface::internal(expoly.clone()));

        assert_eq!(collection.len(), 3);
        assert!(!collection.is_empty());

        assert_eq!(collection.top_surfaces().len(), 1);
        assert_eq!(collection.bottom_surfaces().len(), 1);
        assert_eq!(collection.solid_surfaces().len(), 2); // top + bottom

        assert!(collection.has_type(SurfaceType::Top));
        assert!(collection.has_type(SurfaceType::Bottom));
        assert!(collection.has_type(SurfaceType::Internal));
        assert!(!collection.has_type(SurfaceType::InternalBridge));
    }

    #[test]
    fn test_surface_collection_filter() {
        let expoly = make_square_expolygon();

        let mut collection = SurfaceCollection::new();
        collection.push(Surface::top(expoly.clone()));
        collection.push(Surface::top(expoly.clone()));
        collection.push(Surface::bottom(expoly.clone()));

        let tops = collection.filter_by_type(SurfaceType::Top);
        assert_eq!(tops.len(), 2);

        let bottoms = collection.filter_by_type(SurfaceType::Bottom);
        assert_eq!(bottoms.len(), 1);
    }

    #[test]
    fn test_surface_type_name() {
        assert_eq!(SurfaceType::Top.name(), "top");
        assert_eq!(SurfaceType::Bottom.name(), "bottom");
        assert_eq!(SurfaceType::BottomBridge.name(), "bottom bridge");
        assert_eq!(SurfaceType::InternalSolid.name(), "internal solid");
        assert_eq!(SurfaceType::Internal.name(), "internal");
        assert_eq!(SurfaceType::InternalBridge.name(), "internal bridge");
        assert_eq!(SurfaceType::InternalVoid.name(), "internal void");
    }
}
