//! Extrusion junction for variable-width perimeters.
//!
//! This module provides the `ExtrusionJunction` type which represents a single
//! point along a variable-width extrusion path, including position and width
//! information.
//!
//! # BambuStudio Reference
//!
//! This corresponds to `src/libslic3r/Arachne/utils/ExtrusionJunction.hpp`

use crate::geometry::Point;
use crate::{unscale, Coord, CoordF};

/// A junction (vertex) in a variable-width extrusion path.
///
/// Each junction specifies a position and the extrusion width at that point.
/// The width can vary along the path, allowing for adaptive wall thickness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtrusionJunction {
    /// The position of the centerline at this junction (in scaled coordinates).
    pub position: Point,

    /// The extrusion width at this junction (in scaled coordinates).
    /// This determines how wide the extruded material will be at this point.
    pub width: Coord,

    /// Which perimeter/wall index this junction belongs to.
    /// Perimeters are counted from outside inwards (0 = outermost wall).
    pub perimeter_index: usize,
}

impl ExtrusionJunction {
    /// Create a new extrusion junction.
    ///
    /// # Arguments
    /// * `position` - The XY position of the junction (scaled coordinates)
    /// * `width` - The extrusion width at this point (scaled coordinates)
    /// * `perimeter_index` - Which perimeter this belongs to (0 = outer)
    pub fn new(position: Point, width: Coord, perimeter_index: usize) -> Self {
        Self {
            position,
            width,
            perimeter_index,
        }
    }

    /// Create a junction with a specific width in millimeters.
    pub fn with_width_mm(position: Point, width_mm: CoordF, perimeter_index: usize) -> Self {
        Self {
            position,
            width: crate::scale(width_mm),
            perimeter_index,
        }
    }

    /// Get the X coordinate (scaled).
    #[inline]
    pub fn x(&self) -> Coord {
        self.position.x
    }

    /// Get the Y coordinate (scaled).
    #[inline]
    pub fn y(&self) -> Coord {
        self.position.y
    }

    /// Get the extrusion width in millimeters.
    #[inline]
    pub fn width_mm(&self) -> CoordF {
        unscale(self.width)
    }

    /// Get the position as a Point.
    #[inline]
    pub fn point(&self) -> Point {
        self.position
    }

    /// Calculate the distance to another junction.
    pub fn distance_to(&self, other: &ExtrusionJunction) -> CoordF {
        self.position.distance(&other.position)
    }

    /// Calculate the squared distance to another junction.
    /// Returns i128 to avoid overflow with large coordinates.
    pub fn distance_squared_to(&self, other: &ExtrusionJunction) -> i128 {
        self.position.distance_squared(&other.position)
    }

    /// Check if this junction has the same position as another (within tolerance).
    pub fn coincides_with(&self, other: &ExtrusionJunction, tolerance: Coord) -> bool {
        self.distance_squared_to(other) <= (tolerance as i128) * (tolerance as i128)
    }

    /// Linear interpolation between two junctions.
    ///
    /// Returns a new junction at parameter t (0.0 = self, 1.0 = other).
    pub fn lerp(&self, other: &ExtrusionJunction, t: f64) -> ExtrusionJunction {
        let x = self.position.x as f64 + t * (other.position.x - self.position.x) as f64;
        let y = self.position.y as f64 + t * (other.position.y - self.position.y) as f64;
        let w = self.width as f64 + t * (other.width - self.width) as f64;

        ExtrusionJunction {
            position: Point::new(x.round() as Coord, y.round() as Coord),
            width: w.round() as Coord,
            perimeter_index: self.perimeter_index,
        }
    }

    /// Check if this junction is an external (outer) perimeter.
    #[inline]
    pub fn is_external(&self) -> bool {
        self.perimeter_index == 0
    }
}

impl From<(Point, Coord, usize)> for ExtrusionJunction {
    fn from((position, width, perimeter_index): (Point, Coord, usize)) -> Self {
        Self::new(position, width, perimeter_index)
    }
}

/// A collection of extrusion junctions forming a path segment.
pub type ExtrusionJunctions = Vec<ExtrusionJunction>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scale;

    #[test]
    fn test_junction_new() {
        let p = Point::new(scale(10.0), scale(20.0));
        let j = ExtrusionJunction::new(p, scale(0.45), 0);

        assert_eq!(j.x(), scale(10.0));
        assert_eq!(j.y(), scale(20.0));
        assert!((j.width_mm() - 0.45).abs() < 0.001);
        assert_eq!(j.perimeter_index, 0);
        assert!(j.is_external());
    }

    #[test]
    fn test_junction_with_width_mm() {
        let p = Point::new(scale(5.0), scale(5.0));
        let j = ExtrusionJunction::with_width_mm(p, 0.4, 1);

        assert!((j.width_mm() - 0.4).abs() < 0.001);
        assert!(!j.is_external());
    }

    #[test]
    fn test_junction_distance() {
        let j1 = ExtrusionJunction::new(Point::new(0, 0), scale(0.4), 0);
        let j2 = ExtrusionJunction::new(Point::new(scale(3.0), scale(4.0)), scale(0.4), 0);

        // distance_to returns scaled distance, so we need to unscale for mm
        let dist = j1.distance_to(&j2);
        let dist_mm = unscale(dist as Coord);
        assert!((dist_mm - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_junction_lerp() {
        let j1 = ExtrusionJunction::new(Point::new(0, 0), scale(0.3), 0);
        let j2 = ExtrusionJunction::new(Point::new(scale(10.0), scale(10.0)), scale(0.5), 0);

        let mid = j1.lerp(&j2, 0.5);

        assert!((unscale(mid.x()) - 5.0).abs() < 0.001);
        assert!((unscale(mid.y()) - 5.0).abs() < 0.001);
        assert!((mid.width_mm() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_junction_coincides() {
        let j1 = ExtrusionJunction::new(Point::new(scale(1.0), scale(1.0)), scale(0.4), 0);
        let j2 = ExtrusionJunction::new(Point::new(scale(1.005), scale(1.005)), scale(0.4), 0);
        let j3 = ExtrusionJunction::new(Point::new(scale(2.0), scale(2.0)), scale(0.4), 0);

        // j1 and j2 are very close (within 0.01mm)
        assert!(j1.coincides_with(&j2, scale(0.01)));
        // j1 and j3 are far apart
        assert!(!j1.coincides_with(&j3, scale(0.01)));
    }

    #[test]
    fn test_junction_from_tuple() {
        let j: ExtrusionJunction = (Point::new(100, 200), 450000, 2).into();
        assert_eq!(j.x(), 100);
        assert_eq!(j.y(), 200);
        assert_eq!(j.width, 450000);
        assert_eq!(j.perimeter_index, 2);
    }
}
