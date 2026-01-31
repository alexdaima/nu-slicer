//! Print region configuration.
//!
//! This module provides the PrintRegionConfig type for controlling
//! region-specific print settings, mirroring BambuStudio's PrintRegionConfig.

use crate::config::print_config::{InfillPattern, SeamPosition};
use crate::CoordF;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Configuration for a specific print region.
///
/// A region is a section of a print object that may have different
/// settings from other parts (e.g., different infill, perimeters, etc.).
/// This allows for per-region customization within a single object.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrintRegionConfig {
    // === Perimeters ===
    /// Number of perimeters/shells.
    pub perimeters: u32,

    /// External perimeter extrusion width (mm, 0 = auto).
    pub external_perimeter_extrusion_width: CoordF,

    /// Perimeter extrusion width (mm, 0 = auto).
    pub perimeter_extrusion_width: CoordF,

    /// External perimeter speed (mm/s).
    pub external_perimeter_speed: CoordF,

    /// Perimeter speed (mm/s).
    pub perimeter_speed: CoordF,

    /// Small perimeter speed (mm/s).
    pub small_perimeter_speed: CoordF,

    /// Enable thin walls detection.
    pub thin_walls: bool,

    /// Enable detect bridging perimeters.
    pub overhangs: bool,

    /// Extra perimeters if needed for vertical shells.
    pub extra_perimeters: bool,

    /// Extra perimeters on overhangs.
    pub extra_perimeters_on_overhangs: bool,

    // === Infill ===
    /// Infill density (0.0 - 1.0).
    pub fill_density: CoordF,

    /// Infill pattern.
    pub fill_pattern: InfillPattern,

    /// Solid infill pattern (for top/bottom).
    pub solid_fill_pattern: InfillPattern,

    /// Top solid infill pattern.
    pub top_fill_pattern: InfillPattern,

    /// Bottom solid infill pattern.
    pub bottom_fill_pattern: InfillPattern,

    /// Infill angle (degrees).
    pub fill_angle: CoordF,

    /// Infill extrusion width (mm, 0 = auto).
    pub infill_extrusion_width: CoordF,

    /// Solid infill extrusion width (mm, 0 = auto).
    pub solid_infill_extrusion_width: CoordF,

    /// Top solid infill extrusion width (mm, 0 = auto).
    pub top_infill_extrusion_width: CoordF,

    /// Infill speed (mm/s).
    pub infill_speed: CoordF,

    /// Solid infill speed (mm/s).
    pub solid_infill_speed: CoordF,

    /// Top solid infill speed (mm/s).
    pub top_solid_infill_speed: CoordF,

    /// Infill overlap with perimeters (ratio, 0.0 - 1.0).
    pub infill_overlap: CoordF,

    /// Infill anchor length (mm).
    pub infill_anchor: CoordF,

    /// Maximum infill anchor length (mm).
    pub infill_anchor_max: CoordF,

    // === Solid Layers ===
    /// Number of solid top layers.
    pub top_solid_layers: u32,

    /// Number of solid bottom layers.
    pub bottom_solid_layers: u32,

    /// Minimum shell thickness (mm) for solid infill.
    pub top_solid_min_thickness: CoordF,

    /// Minimum shell thickness (mm) for solid infill.
    pub bottom_solid_min_thickness: CoordF,

    // === Bridges ===
    /// Bridge speed (mm/s).
    pub bridge_speed: CoordF,

    /// Bridge flow ratio.
    pub bridge_flow_ratio: CoordF,

    /// Bridge angle (degrees, 0 = auto).
    pub bridge_angle: CoordF,

    // === Gap Fill ===
    /// Enable gap fill.
    pub gap_fill_enabled: bool,

    /// Gap fill speed (mm/s).
    pub gap_fill_speed: CoordF,

    // === Seam ===
    /// Seam position preference.
    pub seam_position: SeamPosition,

    /// Seam angle cost (for seam placement algorithm).
    pub seam_angle_cost: CoordF,

    /// Seam travel cost (for seam placement algorithm).
    pub seam_travel_cost: CoordF,

    // === Ironing ===
    /// Enable ironing (smoothing top surfaces).
    pub ironing: bool,

    /// Ironing type.
    pub ironing_type: IroningType,

    /// Ironing flow rate ratio.
    pub ironing_flow_rate: CoordF,

    /// Ironing spacing (mm).
    pub ironing_spacing: CoordF,

    /// Ironing speed (mm/s).
    pub ironing_speed: CoordF,

    // === Fuzzy Skin ===
    /// Enable fuzzy skin.
    pub fuzzy_skin: bool,

    /// Fuzzy skin mode.
    pub fuzzy_skin_mode: FuzzySkinMode,

    /// Fuzzy skin thickness (mm).
    pub fuzzy_skin_thickness: CoordF,

    /// Fuzzy skin point distance (mm).
    pub fuzzy_skin_point_distance: CoordF,

    // === Misc ===
    /// Region identifier/name.
    pub region_id: usize,

    /// Extruder index for this region (0-based).
    pub extruder: usize,

    /// Infill extruder index (0 = same as perimeter extruder).
    pub infill_extruder: usize,

    /// Solid infill extruder index.
    pub solid_infill_extruder: usize,
}

impl PrintRegionConfig {
    /// Create a new PrintRegionConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config with a specific region ID.
    pub fn with_region_id(region_id: usize) -> Self {
        Self {
            region_id,
            ..Default::default()
        }
    }

    /// Builder method: set number of perimeters.
    pub fn perimeters(mut self, count: u32) -> Self {
        self.perimeters = count;
        self
    }

    /// Builder method: set infill density.
    pub fn fill_density(mut self, density: CoordF) -> Self {
        self.fill_density = density;
        self
    }

    /// Builder method: set infill pattern.
    pub fn fill_pattern(mut self, pattern: InfillPattern) -> Self {
        self.fill_pattern = pattern;
        self
    }

    /// Builder method: set top solid layers.
    pub fn top_solid_layers(mut self, layers: u32) -> Self {
        self.top_solid_layers = layers;
        self
    }

    /// Builder method: set bottom solid layers.
    pub fn bottom_solid_layers(mut self, layers: u32) -> Self {
        self.bottom_solid_layers = layers;
        self
    }

    /// Builder method: set extruder.
    pub fn extruder(mut self, extruder: usize) -> Self {
        self.extruder = extruder;
        self
    }

    /// Builder method: enable/disable ironing.
    pub fn ironing(mut self, enabled: bool) -> Self {
        self.ironing = enabled;
        self
    }

    /// Builder method: enable/disable fuzzy skin.
    pub fn fuzzy_skin(mut self, enabled: bool) -> Self {
        self.fuzzy_skin = enabled;
        self
    }

    /// Get the effective infill extruder (falls back to perimeter extruder).
    pub fn effective_infill_extruder(&self) -> usize {
        if self.infill_extruder > 0 {
            self.infill_extruder
        } else {
            self.extruder
        }
    }

    /// Get the effective solid infill extruder.
    pub fn effective_solid_infill_extruder(&self) -> usize {
        if self.solid_infill_extruder > 0 {
            self.solid_infill_extruder
        } else {
            self.effective_infill_extruder()
        }
    }

    /// Check if this region has sparse infill.
    pub fn has_sparse_infill(&self) -> bool {
        self.fill_density > 0.0 && self.fill_density < 1.0
    }

    /// Check if this region has solid infill (100% density).
    pub fn is_solid(&self) -> bool {
        self.fill_density >= 1.0
    }

    /// Check if this region has no infill.
    pub fn is_hollow(&self) -> bool {
        self.fill_density == 0.0
    }
}

impl Default for PrintRegionConfig {
    fn default() -> Self {
        Self {
            // Perimeters
            perimeters: 3,
            external_perimeter_extrusion_width: 0.0,
            perimeter_extrusion_width: 0.0,
            external_perimeter_speed: 25.0,
            perimeter_speed: 45.0,
            small_perimeter_speed: 25.0,
            thin_walls: true,
            overhangs: true,
            extra_perimeters: true,
            extra_perimeters_on_overhangs: false,

            // Infill
            fill_density: 0.2,
            fill_pattern: InfillPattern::Grid,
            solid_fill_pattern: InfillPattern::Rectilinear,
            top_fill_pattern: InfillPattern::Rectilinear,
            bottom_fill_pattern: InfillPattern::Rectilinear,
            fill_angle: 45.0,
            infill_extrusion_width: 0.0,
            solid_infill_extrusion_width: 0.0,
            top_infill_extrusion_width: 0.0,
            infill_speed: 80.0,
            solid_infill_speed: 40.0,
            top_solid_infill_speed: 30.0,
            infill_overlap: 0.25,
            infill_anchor: 2.5,
            infill_anchor_max: 12.0,

            // Solid Layers
            top_solid_layers: 4,
            bottom_solid_layers: 3,
            top_solid_min_thickness: 0.0,
            bottom_solid_min_thickness: 0.0,

            // Bridges
            bridge_speed: 25.0,
            bridge_flow_ratio: 1.0,
            bridge_angle: 0.0,

            // Gap Fill
            gap_fill_enabled: true,
            gap_fill_speed: 20.0,

            // Seam
            seam_position: SeamPosition::Aligned,
            seam_angle_cost: 1.0,
            seam_travel_cost: 1.0,

            // Ironing
            ironing: false,
            ironing_type: IroningType::TopSurfaces,
            ironing_flow_rate: 0.15,
            ironing_spacing: 0.1,
            ironing_speed: 15.0,

            // Fuzzy Skin
            fuzzy_skin: false,
            fuzzy_skin_mode: FuzzySkinMode::None,
            fuzzy_skin_thickness: 0.3,
            fuzzy_skin_point_distance: 0.8,

            // Misc
            region_id: 0,
            extruder: 0,
            infill_extruder: 0,
            solid_infill_extruder: 0,
        }
    }
}

impl fmt::Display for PrintRegionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrintRegionConfig(region={}, perimeters={}, infill={:.0}%)",
            self.region_id,
            self.perimeters,
            self.fill_density * 100.0
        )
    }
}

/// Ironing type - which surfaces to iron.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IroningType {
    /// Iron all top surfaces.
    #[default]
    TopSurfaces,
    /// Iron only the topmost surface.
    TopmostOnly,
    /// Iron all solid surfaces.
    AllSolid,
}

/// Fuzzy skin mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FuzzySkinMode {
    /// No fuzzy skin.
    #[default]
    None,
    /// Fuzzy skin on external perimeters only.
    External,
    /// Fuzzy skin on all perimeters.
    All,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_region_config_default() {
        let config = PrintRegionConfig::default();
        assert_eq!(config.perimeters, 3);
        assert!((config.fill_density - 0.2).abs() < 1e-6);
        assert_eq!(config.fill_pattern, InfillPattern::Grid);
        assert_eq!(config.top_solid_layers, 4);
        assert_eq!(config.bottom_solid_layers, 3);
    }

    #[test]
    fn test_print_region_config_builder() {
        let config = PrintRegionConfig::new()
            .perimeters(5)
            .fill_density(0.4)
            .fill_pattern(InfillPattern::Gyroid)
            .top_solid_layers(6)
            .extruder(1);

        assert_eq!(config.perimeters, 5);
        assert!((config.fill_density - 0.4).abs() < 1e-6);
        assert_eq!(config.fill_pattern, InfillPattern::Gyroid);
        assert_eq!(config.top_solid_layers, 6);
        assert_eq!(config.extruder, 1);
    }

    #[test]
    fn test_print_region_config_with_region_id() {
        let config = PrintRegionConfig::with_region_id(5);
        assert_eq!(config.region_id, 5);
    }

    #[test]
    fn test_effective_extruders() {
        let mut config = PrintRegionConfig::default();
        config.extruder = 0;
        config.infill_extruder = 0;
        config.solid_infill_extruder = 0;

        assert_eq!(config.effective_infill_extruder(), 0);
        assert_eq!(config.effective_solid_infill_extruder(), 0);

        config.infill_extruder = 2;
        assert_eq!(config.effective_infill_extruder(), 2);
        assert_eq!(config.effective_solid_infill_extruder(), 2);

        config.solid_infill_extruder = 3;
        assert_eq!(config.effective_solid_infill_extruder(), 3);
    }

    #[test]
    fn test_infill_classification() {
        let mut config = PrintRegionConfig::default();

        config.fill_density = 0.0;
        assert!(config.is_hollow());
        assert!(!config.has_sparse_infill());
        assert!(!config.is_solid());

        config.fill_density = 0.5;
        assert!(!config.is_hollow());
        assert!(config.has_sparse_infill());
        assert!(!config.is_solid());

        config.fill_density = 1.0;
        assert!(!config.is_hollow());
        assert!(!config.has_sparse_infill());
        assert!(config.is_solid());
    }

    #[test]
    fn test_ironing_type_default() {
        assert_eq!(IroningType::default(), IroningType::TopSurfaces);
    }

    #[test]
    fn test_fuzzy_skin_mode_default() {
        assert_eq!(FuzzySkinMode::default(), FuzzySkinMode::None);
    }
}
