//! Perimeter generation module.
//!
//! This module handles the generation of perimeters (walls) for each layer,
//! including support for multi-wall generation using polygon offset operations.
//!
//! # Overview
//!
//! Perimeters are the outlines that define the shape of each layer. They are
//! generated from the slice contours by offsetting inward multiple times:
//!
//! - Outer perimeters (external walls) - printed first or last depending on settings
//! - Inner perimeters (internal walls) - printed between outer and infill
//!
//! # Algorithm
//!
//! 1. Start with the slice ExPolygons
//! 2. Offset inward by half the extrusion width to get the outer perimeter centerline
//! 3. Offset inward by the full extrusion width for each subsequent perimeter
//! 4. Continue until the desired number of perimeters is reached or the polygon disappears
//!
//! # Variable-Width Perimeters (Arachne)
//!
//! The `arachne` submodule provides variable-width perimeter generation that
//! adapts the extrusion width based on local geometry. This improves print
//! quality for thin walls and narrow features.
//!
//! # BambuStudio Reference
//!
//! This module corresponds to:
//! - `src/libslic3r/PerimeterGenerator.cpp`
//! - `src/libslic3r/Arachne/` (variable-width perimeters)

pub mod arachne;
pub mod fuzzy_skin;

use crate::clipper::{shrink, OffsetJoinType};
use crate::geometry::{ExPolygon, ExPolygons, Polygon, Polyline};
use crate::CoordF;

/// Configuration for perimeter generation.
#[derive(Debug, Clone)]
pub struct PerimeterConfig {
    /// Number of perimeter loops to generate.
    pub perimeter_count: usize,

    /// Extrusion width for perimeters (mm).
    pub perimeter_extrusion_width: CoordF,

    /// Extrusion width for external (outer) perimeters (mm).
    pub external_perimeter_extrusion_width: CoordF,

    /// Whether to print external perimeters first (outside-in).
    pub external_perimeters_first: bool,

    /// Minimum area for a perimeter loop to be kept (mm²).
    pub min_perimeter_area: CoordF,

    /// Gap fill threshold - fill gaps smaller than this (mm).
    pub gap_fill_threshold: CoordF,

    /// Whether to detect and handle thin walls.
    pub thin_walls: bool,

    /// Join type for perimeter offset corners.
    pub join_type: OffsetJoinType,
}

impl Default for PerimeterConfig {
    fn default() -> Self {
        Self {
            perimeter_count: 3,
            perimeter_extrusion_width: 0.45,
            external_perimeter_extrusion_width: 0.45,
            external_perimeters_first: false,
            min_perimeter_area: 0.01, // 0.01 mm²
            gap_fill_threshold: 0.0,  // Disabled by default
            thin_walls: false,        // Disabled by default
            join_type: OffsetJoinType::Miter,
        }
    }
}

/// A single perimeter loop with associated metadata.
#[derive(Debug, Clone)]
pub struct PerimeterLoop {
    /// The polygon representing the perimeter centerline.
    pub polygon: Polygon,

    /// Whether this is an external (outer) perimeter.
    pub is_external: bool,

    /// Whether this is a contour (outer boundary) or hole (inner boundary).
    pub is_contour: bool,

    /// The perimeter index (0 = outermost, increasing inward).
    pub perimeter_index: usize,

    /// Extrusion width for this perimeter (mm).
    pub extrusion_width: CoordF,

    /// Depth/nesting level (for ordering).
    pub depth: usize,
}

impl PerimeterLoop {
    /// Create a new perimeter loop.
    pub fn new(
        polygon: Polygon,
        is_external: bool,
        is_contour: bool,
        perimeter_index: usize,
        extrusion_width: CoordF,
    ) -> Self {
        Self {
            polygon,
            is_external,
            is_contour,
            perimeter_index,
            extrusion_width,
            depth: 0,
        }
    }

    /// Get the perimeter length in mm.
    pub fn length_mm(&self) -> CoordF {
        // perimeter() already returns unscaled CoordF in mm
        self.polygon.perimeter() / crate::SCALING_FACTOR
    }

    /// Convert to a polyline (open path) by splitting at the best seam point.
    pub fn to_polyline(&self) -> Polyline {
        self.polygon.to_polyline()
    }

    /// Convert to a closed polyline.
    pub fn to_closed_polyline(&self) -> Polyline {
        self.polygon.to_closed_polyline()
    }
}

/// Result of perimeter generation for a single region.
#[derive(Debug, Clone, Default)]
pub struct PerimeterResult {
    /// Generated perimeter loops, ordered for printing.
    pub perimeters: Vec<PerimeterLoop>,

    /// The area remaining after all perimeters (for infill).
    pub infill_area: ExPolygons,

    /// Thin fill paths (areas too narrow for perimeters).
    pub thin_fills: Vec<Polyline>,

    /// Gap fill paths (small gaps between perimeters).
    pub gap_fills: Vec<Polyline>,
}

impl PerimeterResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any perimeters were generated.
    pub fn has_perimeters(&self) -> bool {
        !self.perimeters.is_empty()
    }

    /// Get the total number of perimeter loops.
    pub fn perimeter_count(&self) -> usize {
        self.perimeters.len()
    }

    /// Get external perimeters only.
    pub fn external_perimeters(&self) -> impl Iterator<Item = &PerimeterLoop> {
        self.perimeters.iter().filter(|p| p.is_external)
    }

    /// Get internal perimeters only.
    pub fn internal_perimeters(&self) -> impl Iterator<Item = &PerimeterLoop> {
        self.perimeters.iter().filter(|p| !p.is_external)
    }

    /// Get perimeters ordered for outside-in printing.
    pub fn ordered_outside_in(&self) -> Vec<&PerimeterLoop> {
        let mut result: Vec<_> = self.perimeters.iter().collect();
        result.sort_by_key(|p| p.perimeter_index);
        result
    }

    /// Get perimeters ordered for inside-out printing.
    pub fn ordered_inside_out(&self) -> Vec<&PerimeterLoop> {
        let mut result: Vec<_> = self.perimeters.iter().collect();
        result.sort_by_key(|p| std::cmp::Reverse(p.perimeter_index));
        result
    }

    /// Get the total perimeter length in mm.
    pub fn total_length_mm(&self) -> CoordF {
        self.perimeters.iter().map(|p| p.length_mm()).sum()
    }
}

/// Perimeter generator.
///
/// Generates perimeter loops from slice ExPolygons using polygon offset operations.
#[derive(Debug, Clone)]
pub struct PerimeterGenerator {
    config: PerimeterConfig,
}

impl PerimeterGenerator {
    /// Create a new perimeter generator with the given configuration.
    pub fn new(config: PerimeterConfig) -> Self {
        Self { config }
    }

    /// Create a perimeter generator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PerimeterConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &PerimeterConfig {
        &self.config
    }

    /// Get mutable access to the configuration.
    pub fn config_mut(&mut self) -> &mut PerimeterConfig {
        &mut self.config
    }

    /// Generate perimeters for the given slice ExPolygons.
    ///
    /// # Arguments
    /// * `slices` - The slice ExPolygons to generate perimeters for
    ///
    /// # Returns
    /// A PerimeterResult containing the generated perimeters and infill area.
    pub fn generate(&self, slices: &[ExPolygon]) -> PerimeterResult {
        let mut result = PerimeterResult::new();

        if slices.is_empty() || self.config.perimeter_count == 0 {
            result.infill_area = slices.to_vec();
            return result;
        }

        // Track all perimeters by level for ordering
        let mut perimeter_levels: Vec<Vec<PerimeterLoop>> =
            vec![Vec::new(); self.config.perimeter_count];

        // Current working area (starts as the full slice)
        // Sort slices deterministically by bounding box for reproducible results
        let mut current_area = slices.to_vec();
        Self::sort_expolygons_deterministic(&mut current_area);

        // Generate each perimeter level
        for perimeter_idx in 0..self.config.perimeter_count {
            let is_external = perimeter_idx == 0;

            // Calculate the offset distance for this perimeter
            let extrusion_width = if is_external {
                self.config.external_perimeter_extrusion_width
            } else {
                self.config.perimeter_extrusion_width
            };

            // First perimeter: offset by half width to get centerline at the edge
            // Subsequent perimeters: offset by full width
            let offset_distance = if perimeter_idx == 0 {
                extrusion_width / 2.0
            } else {
                extrusion_width
            };

            // Shrink the current area to get the perimeter centerlines
            let perimeter_area = shrink(&current_area, offset_distance, self.config.join_type);

            if perimeter_area.is_empty() {
                // No more room for perimeters
                break;
            }

            // Extract perimeter loops from the offset result
            for expoly in &perimeter_area {
                // Add contour as a perimeter
                if !expoly.contour.is_empty() && self.is_valid_perimeter(&expoly.contour) {
                    let loop_item = PerimeterLoop::new(
                        expoly.contour.clone(),
                        is_external,
                        true, // is_contour
                        perimeter_idx,
                        extrusion_width,
                    );
                    perimeter_levels[perimeter_idx].push(loop_item);
                }

                // Add holes as perimeters (they're inner boundaries)
                // Sort holes deterministically
                let mut sorted_holes = expoly.holes.clone();
                sorted_holes.sort_by(|a, b| Self::compare_polygons_deterministic(a, b));

                for hole in &sorted_holes {
                    if !hole.is_empty() && self.is_valid_perimeter(hole) {
                        let loop_item = PerimeterLoop::new(
                            hole.clone(),
                            is_external,
                            false, // is_contour (it's a hole)
                            perimeter_idx,
                            extrusion_width,
                        );
                        perimeter_levels[perimeter_idx].push(loop_item);
                    }
                }
            }

            // Update current area for next iteration
            current_area = perimeter_area;
        }

        // The remaining area after all perimeters is the infill area
        // We need to shrink once more by half the inner perimeter width
        let inner_offset = self.config.perimeter_extrusion_width / 2.0;
        result.infill_area = shrink(&current_area, inner_offset, self.config.join_type);

        // Order perimeters for printing
        // Default is inside-out (external perimeters last) unless configured otherwise
        // Sort each level deterministically before adding
        if self.config.external_perimeters_first {
            // Outside-in: start from perimeter 0 (external)
            for mut level in perimeter_levels {
                Self::sort_perimeter_loops_deterministic(&mut level);
                result.perimeters.extend(level);
            }
        } else {
            // Inside-out: start from innermost perimeter
            for mut level in perimeter_levels.into_iter().rev() {
                Self::sort_perimeter_loops_deterministic(&mut level);
                result.perimeters.extend(level);
            }
        }

        result
    }

    /// Generate perimeters for a single ExPolygon.
    pub fn generate_single(&self, expoly: &ExPolygon) -> PerimeterResult {
        self.generate(&[expoly.clone()][..])
    }

    /// Sort ExPolygons deterministically by bounding box (min X, then min Y, then area).
    fn sort_expolygons_deterministic(expolys: &mut [ExPolygon]) {
        expolys.sort_by(|a, b| {
            let bb_a = a.bounding_box();
            let bb_b = b.bounding_box();

            // Compare by min X first
            match bb_a.min.x.cmp(&bb_b.min.x) {
                std::cmp::Ordering::Equal => {}
                other => return other,
            }

            // Then by min Y
            match bb_a.min.y.cmp(&bb_b.min.y) {
                std::cmp::Ordering::Equal => {}
                other => return other,
            }

            // Finally by area (larger first for stability)
            let area_a = a.area().abs();
            let area_b = b.area().abs();
            area_b
                .partial_cmp(&area_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Compare two polygons deterministically for sorting.
    fn compare_polygons_deterministic(a: &Polygon, b: &Polygon) -> std::cmp::Ordering {
        let bb_a = a.bounding_box();
        let bb_b = b.bounding_box();

        // Compare by min X first
        match bb_a.min.x.cmp(&bb_b.min.x) {
            std::cmp::Ordering::Equal => {}
            other => return other,
        }

        // Then by min Y
        match bb_a.min.y.cmp(&bb_b.min.y) {
            std::cmp::Ordering::Equal => {}
            other => return other,
        }

        // Finally by area
        let area_a = a.area().abs();
        let area_b = b.area().abs();
        area_b
            .partial_cmp(&area_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Sort perimeter loops deterministically by bounding box.
    fn sort_perimeter_loops_deterministic(loops: &mut [PerimeterLoop]) {
        loops.sort_by(|a, b| Self::compare_polygons_deterministic(&a.polygon, &b.polygon));
    }

    /// Check if a polygon is valid for use as a perimeter.
    fn is_valid_perimeter(&self, polygon: &Polygon) -> bool {
        if polygon.len() < 3 {
            return false;
        }

        // Check minimum area
        // area() returns scaled area (scaled_units²), convert to mm²
        let area_mm2 = polygon.area() / (crate::SCALING_FACTOR * crate::SCALING_FACTOR);
        if area_mm2.abs() < self.config.min_perimeter_area {
            return false;
        }

        true
    }

    /// Calculate the inset distance for getting to the infill boundary.
    ///
    /// This is the total distance from the slice edge to the inner infill boundary.
    pub fn total_inset_distance(&self) -> CoordF {
        if self.config.perimeter_count == 0 {
            return 0.0;
        }

        // First perimeter: half external width
        let mut total = self.config.external_perimeter_extrusion_width / 2.0;

        // Inner perimeters: full width each
        if self.config.perimeter_count > 1 {
            total +=
                self.config.perimeter_extrusion_width * (self.config.perimeter_count - 1) as CoordF;
        }

        // Half width on the inside
        total += self.config.perimeter_extrusion_width / 2.0;

        total
    }
}

impl Default for PerimeterGenerator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Compute the infill area for given slices and perimeter configuration.
///
/// This is a convenience function that generates perimeters and returns only the infill area.
pub fn compute_infill_area(slices: &[ExPolygon], config: &PerimeterConfig) -> ExPolygons {
    let generator = PerimeterGenerator::new(config.clone());
    let result = generator.generate(slices);
    result.infill_area
}

/// Generate perimeters with default configuration.
pub fn generate_perimeters(slices: &[ExPolygon]) -> PerimeterResult {
    PerimeterGenerator::with_defaults().generate(slices)
}

/// Generate perimeters with custom parameters.
pub fn generate_perimeters_with(
    slices: &[ExPolygon],
    perimeter_count: usize,
    extrusion_width: CoordF,
) -> PerimeterResult {
    let config = PerimeterConfig {
        perimeter_count,
        perimeter_extrusion_width: extrusion_width,
        external_perimeter_extrusion_width: extrusion_width,
        ..Default::default()
    };
    PerimeterGenerator::new(config).generate(slices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Point;

    fn make_square_mm(x: f64, y: f64, size: f64) -> ExPolygon {
        let poly = Polygon::rectangle(
            Point::new(crate::scale(x), crate::scale(y)),
            Point::new(crate::scale(x + size), crate::scale(y + size)),
        );
        poly.into()
    }

    fn make_square_with_hole_mm(
        x: f64,
        y: f64,
        outer_size: f64,
        hole_offset: f64,
        hole_size: f64,
    ) -> ExPolygon {
        let outer = Polygon::rectangle(
            Point::new(crate::scale(x), crate::scale(y)),
            Point::new(crate::scale(x + outer_size), crate::scale(y + outer_size)),
        );
        let inner = Polygon::rectangle(
            Point::new(crate::scale(x + hole_offset), crate::scale(y + hole_offset)),
            Point::new(
                crate::scale(x + hole_offset + hole_size),
                crate::scale(y + hole_offset + hole_size),
            ),
        );
        ExPolygon::with_holes(outer, vec![inner])
    }

    #[test]
    fn test_perimeter_config_default() {
        let config = PerimeterConfig::default();
        assert_eq!(config.perimeter_count, 3);
        assert!((config.perimeter_extrusion_width - 0.45).abs() < 1e-6);
    }

    #[test]
    fn test_generator_simple_square() {
        let square = make_square_mm(0.0, 0.0, 20.0);
        let config = PerimeterConfig {
            perimeter_count: 2,
            perimeter_extrusion_width: 0.5,
            external_perimeter_extrusion_width: 0.5,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let result = generator.generate(&[square]);

        // Should have perimeters
        assert!(result.has_perimeters());

        // Should have external perimeters
        assert!(result.external_perimeters().count() > 0);

        // Should have infill area
        assert!(!result.infill_area.is_empty());

        println!("Generated {} perimeters", result.perimeter_count());
        println!("Total perimeter length: {:.2} mm", result.total_length_mm());
    }

    #[test]
    fn test_generator_with_hole() {
        let expoly = make_square_with_hole_mm(0.0, 0.0, 30.0, 10.0, 10.0);
        let config = PerimeterConfig {
            perimeter_count: 3,
            perimeter_extrusion_width: 0.45,
            external_perimeter_extrusion_width: 0.45,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let result = generator.generate(&[expoly]);

        // Should have perimeters for both outer contour and hole
        assert!(result.has_perimeters());
        assert!(result.perimeter_count() > 3); // Multiple perimeters for contour + hole

        // Should have infill area
        assert!(!result.infill_area.is_empty());
    }

    #[test]
    fn test_generator_too_small() {
        // 1mm x 1mm square with 0.5mm perimeter width and 3 perimeters
        // Total inset would be about 1.5mm, so this should shrink to nothing
        let tiny = make_square_mm(0.0, 0.0, 1.0);
        let config = PerimeterConfig {
            perimeter_count: 3,
            perimeter_extrusion_width: 0.5,
            external_perimeter_extrusion_width: 0.5,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let result = generator.generate(&[tiny]);

        // Should still generate some perimeters (at least the first one)
        // but infill area will likely be empty
        println!(
            "Tiny square: {} perimeters, {} infill areas",
            result.perimeter_count(),
            result.infill_area.len()
        );
    }

    #[test]
    fn test_generator_zero_perimeters() {
        let square = make_square_mm(0.0, 0.0, 20.0);
        let config = PerimeterConfig {
            perimeter_count: 0,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let result = generator.generate(&[square.clone()]);

        // No perimeters
        assert!(!result.has_perimeters());

        // Infill area should be the original slice
        assert!(!result.infill_area.is_empty());
    }

    #[test]
    fn test_ordering_outside_in() {
        let square = make_square_mm(0.0, 0.0, 20.0);
        let config = PerimeterConfig {
            perimeter_count: 3,
            external_perimeters_first: true,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let result = generator.generate(&[square]);

        let ordered = result.ordered_outside_in();
        assert!(!ordered.is_empty());

        // First should be external (index 0)
        assert!(ordered[0].is_external);
        assert_eq!(ordered[0].perimeter_index, 0);
    }

    #[test]
    fn test_ordering_inside_out() {
        let square = make_square_mm(0.0, 0.0, 20.0);
        let config = PerimeterConfig {
            perimeter_count: 3,
            external_perimeters_first: false,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let result = generator.generate(&[square]);

        let ordered = result.ordered_inside_out();
        assert!(!ordered.is_empty());

        // First should be innermost (highest index)
        // Last should be external
        assert!(ordered.last().unwrap().is_external);
    }

    #[test]
    fn test_total_inset_distance() {
        let config = PerimeterConfig {
            perimeter_count: 3,
            perimeter_extrusion_width: 0.5,
            external_perimeter_extrusion_width: 0.5,
            ..Default::default()
        };

        let generator = PerimeterGenerator::new(config);
        let inset = generator.total_inset_distance();

        // Expected: 0.25 (half external) + 0.5 + 0.5 (2 inner) + 0.25 (half inner) = 1.5
        assert!((inset - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_convenience_functions() {
        let square = make_square_mm(0.0, 0.0, 20.0);

        // Test generate_perimeters
        let result = generate_perimeters(&[square.clone()]);
        assert!(result.has_perimeters());

        // Test generate_perimeters_with
        let result2 = generate_perimeters_with(&[square.clone()], 2, 0.4);
        assert!(result2.has_perimeters());

        // Test compute_infill_area
        let config = PerimeterConfig::default();
        let infill = compute_infill_area(&[square], &config);
        assert!(!infill.is_empty());
    }

    #[test]
    fn test_perimeter_loop_to_polyline() {
        let square = Polygon::rectangle(
            Point::new(crate::scale(0.0), crate::scale(0.0)),
            Point::new(crate::scale(10.0), crate::scale(10.0)),
        );

        let loop_item = PerimeterLoop::new(square.clone(), true, true, 0, 0.45);

        let polyline = loop_item.to_polyline();
        assert!(!polyline.is_empty());
        assert_eq!(polyline.len(), square.len());

        let closed = loop_item.to_closed_polyline();
        assert!(!closed.is_empty());
        assert_eq!(closed.len(), square.len() + 1); // Closed has extra point
    }

    #[test]
    fn test_multiple_regions() {
        // Two separate squares
        let square1 = make_square_mm(0.0, 0.0, 15.0);
        let square2 = make_square_mm(20.0, 0.0, 15.0);

        let generator = PerimeterGenerator::with_defaults();
        let result = generator.generate(&[square1, square2]);

        // Should have perimeters for both regions
        assert!(result.perimeter_count() >= 6); // At least 3 per region

        // Should have infill areas for both
        assert!(result.infill_area.len() >= 2);
    }
}
