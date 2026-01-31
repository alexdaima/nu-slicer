//! ExPolygon type for polygons with holes.
//!
//! This module provides the ExPolygon type representing a polygon with holes
//! (exterior contour + interior hole contours), mirroring BambuStudio's ExPolygon class.

use super::{BoundingBox, Point, Polygon, Polyline};
use crate::{Coord, CoordF};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A polygon with holes (exterior polygon + interior hole polygons).
///
/// The contour is the outer boundary (should be counter-clockwise for positive area).
/// The holes are interior boundaries (should be clockwise).
#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExPolygon {
    /// The outer contour of the polygon.
    pub contour: Polygon,
    /// The holes (interior contours) of the polygon.
    pub holes: Vec<Polygon>,
}

impl ExPolygon {
    /// Create a new ExPolygon with only a contour and no holes.
    #[inline]
    pub fn new(contour: Polygon) -> Self {
        Self {
            contour,
            holes: Vec::new(),
        }
    }

    /// Create a new ExPolygon with a contour and holes.
    #[inline]
    pub fn with_holes(contour: Polygon, holes: Vec<Polygon>) -> Self {
        Self { contour, holes }
    }

    /// Create an empty ExPolygon.
    #[inline]
    pub fn empty() -> Self {
        Self {
            contour: Polygon::new(),
            holes: Vec::new(),
        }
    }

    /// Check if the ExPolygon is empty (no contour points).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.contour.is_empty()
    }

    /// Get the number of holes.
    #[inline]
    pub fn hole_count(&self) -> usize {
        self.holes.len()
    }

    /// Check if this ExPolygon has any holes.
    #[inline]
    pub fn has_holes(&self) -> bool {
        !self.holes.is_empty()
    }

    /// Add a hole to the ExPolygon.
    #[inline]
    pub fn add_hole(&mut self, hole: Polygon) {
        self.holes.push(hole);
    }

    /// Clear all holes.
    #[inline]
    pub fn clear_holes(&mut self) {
        self.holes.clear();
    }

    /// Calculate the area of the ExPolygon (contour area minus hole areas).
    pub fn area(&self) -> CoordF {
        let contour_area = self.contour.area();
        let holes_area: CoordF = self.holes.iter().map(|h| h.area()).sum();
        contour_area - holes_area
    }

    /// Calculate the signed area of the ExPolygon.
    pub fn signed_area(&self) -> CoordF {
        let contour_area = self.contour.signed_area();
        let holes_area: CoordF = self.holes.iter().map(|h| h.signed_area().abs()).sum();
        if contour_area >= 0.0 {
            contour_area - holes_area
        } else {
            contour_area + holes_area
        }
    }

    /// Calculate the total perimeter (contour + all holes).
    pub fn perimeter(&self) -> CoordF {
        let contour_perim = self.contour.perimeter();
        let holes_perim: CoordF = self.holes.iter().map(|h| h.perimeter()).sum();
        contour_perim + holes_perim
    }

    /// Get the bounding box of the ExPolygon (same as contour's bounding box).
    #[inline]
    pub fn bounding_box(&self) -> BoundingBox {
        self.contour.bounding_box()
    }

    /// Check if a point is inside the ExPolygon (inside contour and not inside any hole).
    pub fn contains_point(&self, p: &Point) -> bool {
        if !self.contour.contains_point(p) {
            return false;
        }

        // Check that point is not inside any hole
        for hole in &self.holes {
            if hole.contains_point(p) {
                return false;
            }
        }

        true
    }

    /// Check if a point is on the boundary of the ExPolygon.
    pub fn is_point_on_boundary(&self, p: &Point, tolerance: Coord) -> bool {
        if self.contour.is_point_on_boundary(p, tolerance) {
            return true;
        }

        for hole in &self.holes {
            if hole.is_point_on_boundary(p, tolerance) {
                return true;
            }
        }

        false
    }

    /// Get the centroid of the ExPolygon.
    /// This is an approximation that uses the contour's centroid.
    #[inline]
    pub fn centroid(&self) -> Point {
        self.contour.centroid()
    }

    /// Ensure the contour is counter-clockwise and holes are clockwise.
    pub fn make_canonical(&mut self) {
        self.contour.make_counter_clockwise();
        for hole in &mut self.holes {
            hole.make_clockwise();
        }
    }

    /// Check if the ExPolygon has canonical orientation
    /// (contour CCW, holes CW).
    pub fn is_canonical(&self) -> bool {
        if !self.contour.is_counter_clockwise() {
            return false;
        }

        for hole in &self.holes {
            if !hole.is_clockwise() {
                return false;
            }
        }

        true
    }

    /// Translate the ExPolygon by a vector.
    pub fn translate(&mut self, v: Point) {
        self.contour.translate(v);
        for hole in &mut self.holes {
            hole.translate(v);
        }
    }

    /// Return a translated copy of the ExPolygon.
    pub fn translated(&self, v: Point) -> Self {
        let mut result = self.clone();
        result.translate(v);
        result
    }

    /// Scale the ExPolygon about the origin.
    pub fn scale(&mut self, factor: CoordF) {
        self.contour.scale(factor);
        for hole in &mut self.holes {
            hole.scale(factor);
        }
    }

    /// Return a scaled copy of the ExPolygon.
    pub fn scaled(&self, factor: CoordF) -> Self {
        let mut result = self.clone();
        result.scale(factor);
        result
    }

    /// Rotate the ExPolygon about the origin.
    pub fn rotate(&mut self, angle: CoordF) {
        self.contour.rotate(angle);
        for hole in &mut self.holes {
            hole.rotate(angle);
        }
    }

    /// Return a rotated copy of the ExPolygon.
    pub fn rotated(&self, angle: CoordF) -> Self {
        let mut result = self.clone();
        result.rotate(angle);
        result
    }

    /// Rotate the ExPolygon about a center point.
    pub fn rotate_around(&mut self, angle: CoordF, center: Point) {
        self.contour.rotate_around(angle, center);
        for hole in &mut self.holes {
            hole.rotate_around(angle, center);
        }
    }

    /// Return a copy rotated about a center point.
    pub fn rotated_around(&self, angle: CoordF, center: Point) -> Self {
        let mut result = self.clone();
        result.rotate_around(angle, center);
        result
    }

    /// Simplify the ExPolygon by removing collinear and duplicate points.
    pub fn simplify(&mut self, tolerance: Coord) {
        self.contour.simplify(tolerance);
        for hole in &mut self.holes {
            hole.simplify(tolerance);
        }
        // Remove degenerate holes
        self.holes.retain(|h| h.len() >= 3);
    }

    /// Return a simplified copy of the ExPolygon.
    pub fn simplified(&self, tolerance: Coord) -> Self {
        let mut result = self.clone();
        result.simplify(tolerance);
        result
    }

    /// Check if the ExPolygon is valid.
    pub fn is_valid(&self) -> bool {
        if !self.contour.is_valid() {
            return false;
        }

        for hole in &self.holes {
            if !hole.is_valid() {
                return false;
            }
        }

        true
    }

    /// Get all polygons (contour and holes) as a vector.
    pub fn all_polygons(&self) -> Vec<&Polygon> {
        let mut result = Vec::with_capacity(1 + self.holes.len());
        result.push(&self.contour);
        result.extend(self.holes.iter());
        result
    }

    /// Get all polygons as mutable references.
    pub fn all_polygons_mut(&mut self) -> Vec<&mut Polygon> {
        let mut result = Vec::with_capacity(1 + self.holes.len());
        result.push(&mut self.contour);
        result.extend(self.holes.iter_mut());
        result
    }

    /// Convert to a vector of polylines (contour and holes as open paths).
    pub fn to_polylines(&self) -> Vec<Polyline> {
        let mut result = Vec::with_capacity(1 + self.holes.len());
        result.push(self.contour.to_closed_polyline());
        for hole in &self.holes {
            result.push(hole.to_closed_polyline());
        }
        result
    }

    /// Convert to a vector of polygons (contour and holes).
    pub fn to_polygons(&self) -> Vec<Polygon> {
        let mut result = Vec::with_capacity(1 + self.holes.len());
        result.push(self.contour.clone());
        result.extend(self.holes.iter().cloned());
        result
    }

    /// Create a rectangular ExPolygon.
    pub fn rectangle(min: Point, max: Point) -> Self {
        Self::new(Polygon::rectangle(min, max))
    }

    /// Create a square ExPolygon.
    pub fn square(center: Point, half_size: Coord) -> Self {
        Self::new(Polygon::square(center, half_size))
    }

    /// Create a circular ExPolygon approximation.
    pub fn circle(center: Point, radius: Coord, segments: usize) -> Self {
        Self::new(Polygon::circle(center, radius, segments))
    }

    /// Get the total number of points in the ExPolygon.
    pub fn point_count(&self) -> usize {
        self.contour.len() + self.holes.iter().map(|h| h.len()).sum::<usize>()
    }

    /// Find the closest point on any boundary to the given point.
    pub fn closest_point(&self, p: &Point) -> Point {
        let mut closest = self.contour.closest_point(p);
        let mut min_dist = p.distance_squared(&closest);

        for hole in &self.holes {
            let hole_closest = hole.closest_point(p);
            let dist = p.distance_squared(&hole_closest);
            if dist < min_dist {
                min_dist = dist;
                closest = hole_closest;
            }
        }

        closest
    }

    /// Distance from a point to the nearest boundary.
    pub fn distance_to_point(&self, p: &Point) -> CoordF {
        let closest = self.closest_point(p);
        p.distance(&closest)
    }

    /// Remove holes that are too small.
    pub fn remove_small_holes(&mut self, min_area: CoordF) {
        self.holes.retain(|h| h.area() >= min_area);
    }
}

impl fmt::Debug for ExPolygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExPolygon(contour: {} points, {} holes)",
            self.contour.len(),
            self.holes.len()
        )
    }
}

impl fmt::Display for ExPolygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExPolygon[contour: {}", self.contour)?;
        for (i, hole) in self.holes.iter().enumerate() {
            write!(f, ", hole{}: {}", i, hole)?;
        }
        write!(f, "]")
    }
}

impl From<Polygon> for ExPolygon {
    fn from(polygon: Polygon) -> Self {
        Self::new(polygon)
    }
}

impl From<ExPolygon> for Polygon {
    /// Convert to the contour polygon, discarding holes.
    fn from(expoly: ExPolygon) -> Self {
        expoly.contour
    }
}

/// Type alias for a collection of ExPolygons.
pub type ExPolygons = Vec<ExPolygon>;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_with_hole() -> ExPolygon {
        // Outer square 0-100
        let contour = Polygon::from_points(vec![
            Point::new(0, 0),
            Point::new(100, 0),
            Point::new(100, 100),
            Point::new(0, 100),
        ]);

        // Inner square (hole) 25-75, clockwise
        let hole = Polygon::from_points(vec![
            Point::new(25, 25),
            Point::new(25, 75),
            Point::new(75, 75),
            Point::new(75, 25),
        ]);

        ExPolygon::with_holes(contour, vec![hole])
    }

    #[test]
    fn test_expolygon_new() {
        let contour = Polygon::rectangle(Point::new(0, 0), Point::new(100, 100));
        let expoly = ExPolygon::new(contour);
        assert!(!expoly.is_empty());
        assert!(!expoly.has_holes());
        assert_eq!(expoly.hole_count(), 0);
    }

    #[test]
    fn test_expolygon_with_holes() {
        let expoly = make_square_with_hole();
        assert!(!expoly.is_empty());
        assert!(expoly.has_holes());
        assert_eq!(expoly.hole_count(), 1);
    }

    #[test]
    fn test_expolygon_area() {
        let expoly = make_square_with_hole();
        let area = expoly.area();
        // 100x100 = 10000, minus 50x50 = 2500, equals 7500
        assert!((area - 7500.0).abs() < 1.0);
    }

    #[test]
    fn test_expolygon_perimeter() {
        let expoly = make_square_with_hole();
        let perim = expoly.perimeter();
        // Outer: 400, Inner: 200, Total: 600
        assert!((perim - 600.0).abs() < 1.0);
    }

    #[test]
    fn test_expolygon_bounding_box() {
        let expoly = make_square_with_hole();
        let bb = expoly.bounding_box();
        assert_eq!(bb.min.x, 0);
        assert_eq!(bb.min.y, 0);
        assert_eq!(bb.max.x, 100);
        assert_eq!(bb.max.y, 100);
    }

    #[test]
    fn test_expolygon_contains_point() {
        let expoly = make_square_with_hole();

        // Point inside contour but outside hole
        assert!(expoly.contains_point(&Point::new(10, 10)));
        assert!(expoly.contains_point(&Point::new(90, 90)));

        // Point inside hole
        assert!(!expoly.contains_point(&Point::new(50, 50)));

        // Point outside contour
        assert!(!expoly.contains_point(&Point::new(-10, -10)));
        assert!(!expoly.contains_point(&Point::new(110, 110)));
    }

    #[test]
    fn test_expolygon_translate() {
        let mut expoly = make_square_with_hole();
        expoly.translate(Point::new(10, 20));

        assert_eq!(expoly.contour[0], Point::new(10, 20));
        assert_eq!(expoly.holes[0][0], Point::new(35, 45));
    }

    #[test]
    fn test_expolygon_scale() {
        let mut expoly = make_square_with_hole();
        expoly.scale(2.0);

        assert_eq!(expoly.contour[2], Point::new(200, 200));
        let area = expoly.area();
        // Original area 7500, scaled by 4 = 30000
        assert!((area - 30000.0).abs() < 1.0);
    }

    #[test]
    fn test_expolygon_make_canonical() {
        // Create with wrong orientations
        let contour = Polygon::from_points(vec![
            Point::new(0, 100),
            Point::new(100, 100),
            Point::new(100, 0),
            Point::new(0, 0),
        ]); // Clockwise

        let hole = Polygon::from_points(vec![
            Point::new(25, 25),
            Point::new(75, 25),
            Point::new(75, 75),
            Point::new(25, 75),
        ]); // Counter-clockwise

        let mut expoly = ExPolygon::with_holes(contour, vec![hole]);
        assert!(!expoly.is_canonical());

        expoly.make_canonical();
        assert!(expoly.is_canonical());
    }

    #[test]
    fn test_expolygon_is_valid() {
        let expoly = make_square_with_hole();
        assert!(expoly.is_valid());

        // Invalid: contour with only 2 points
        let invalid = ExPolygon::new(Polygon::from_points(vec![
            Point::new(0, 0),
            Point::new(100, 0),
        ]));
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_expolygon_all_polygons() {
        let expoly = make_square_with_hole();
        let all = expoly.all_polygons();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_expolygon_to_polylines() {
        let expoly = make_square_with_hole();
        let polylines = expoly.to_polylines();
        assert_eq!(polylines.len(), 2);
        // Each polyline should be closed (first point repeated at end)
        assert!(polylines[0].is_closed());
        assert!(polylines[1].is_closed());
    }

    #[test]
    fn test_expolygon_point_count() {
        let expoly = make_square_with_hole();
        assert_eq!(expoly.point_count(), 8); // 4 + 4
    }

    #[test]
    fn test_expolygon_rectangle() {
        let expoly = ExPolygon::rectangle(Point::new(0, 0), Point::new(100, 50));
        assert_eq!(expoly.contour.len(), 4);
        assert!(!expoly.has_holes());
        assert!((expoly.area() - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_expolygon_closest_point() {
        let expoly = make_square_with_hole();

        // Point outside - closest to contour
        let p1 = Point::new(50, -20);
        let closest1 = expoly.closest_point(&p1);
        assert_eq!(closest1.x, 50);
        assert_eq!(closest1.y, 0);

        // Point inside hole - closest to hole boundary
        let p2 = Point::new(50, 50);
        let closest2 = expoly.closest_point(&p2);
        // Should be on one of the hole edges, distance should be 25
        let dist = p2.distance(&closest2);
        assert!((dist - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_expolygon_remove_small_holes() {
        let contour = Polygon::rectangle(Point::new(0, 0), Point::new(100, 100));
        let big_hole = Polygon::rectangle(Point::new(10, 10), Point::new(50, 50)); // area = 1600
        let small_hole = Polygon::rectangle(Point::new(60, 60), Point::new(65, 65)); // area = 25

        let mut expoly = ExPolygon::with_holes(contour, vec![big_hole, small_hole]);
        assert_eq!(expoly.hole_count(), 2);

        expoly.remove_small_holes(100.0);
        assert_eq!(expoly.hole_count(), 1);
    }

    #[test]
    fn test_expolygon_from_polygon() {
        let poly = Polygon::rectangle(Point::new(0, 0), Point::new(100, 100));
        let expoly: ExPolygon = poly.into();
        assert!(!expoly.has_holes());
        assert!((expoly.area() - 10000.0).abs() < 1.0);
    }
}
