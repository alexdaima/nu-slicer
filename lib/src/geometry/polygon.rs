//! Polygon type for closed contours.
//!
//! This module provides the Polygon type representing a closed polygon (boundary),
//! mirroring BambuStudio's Polygon class.

use super::{BoundingBox, Line, Point, Polyline};
use crate::{Coord, CoordF};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// A closed polygon defined by a sequence of points.
///
/// The polygon is implicitly closed - the last point connects back to the first.
/// Points should be ordered counter-clockwise for outer contours (positive area)
/// and clockwise for holes (negative area).
#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Polygon {
    points: Vec<Point>,
}

impl Polygon {
    /// Create a new empty polygon.
    #[inline]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Create a polygon from a vector of points.
    #[inline]
    pub fn from_points(points: Vec<Point>) -> Self {
        Self { points }
    }

    /// Create a polygon with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Get the points of this polygon.
    #[inline]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Get a mutable reference to the points.
    #[inline]
    pub fn points_mut(&mut self) -> &mut Vec<Point> {
        &mut self.points
    }

    /// Consume the polygon and return its points.
    #[inline]
    pub fn into_points(self) -> Vec<Point> {
        self.points
    }

    /// Get the number of points in the polygon.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the polygon is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Add a point to the polygon.
    #[inline]
    pub fn push(&mut self, point: Point) {
        self.points.push(point);
    }

    /// Clear all points from the polygon.
    #[inline]
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Reserve capacity for additional points.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.points.reserve(additional);
    }

    /// Get a point at the given index, wrapping around for indices >= len.
    #[inline]
    pub fn point_at(&self, index: usize) -> Point {
        self.points[index % self.points.len()]
    }

    /// Get the line segment at the given index (from point[i] to point[i+1]).
    #[inline]
    pub fn edge(&self, index: usize) -> Line {
        let len = self.points.len();
        Line::new(self.points[index % len], self.points[(index + 1) % len])
    }

    /// Get all edges of the polygon.
    pub fn edges(&self) -> Vec<Line> {
        if self.points.len() < 2 {
            return Vec::new();
        }

        let mut edges = Vec::with_capacity(self.points.len());
        for i in 0..self.points.len() {
            edges.push(self.edge(i));
        }
        edges
    }

    /// Get the number of edges in the polygon.
    #[inline]
    pub fn edge_count(&self) -> usize {
        if self.points.len() < 2 {
            0
        } else {
            self.points.len()
        }
    }

    /// Calculate the signed area of the polygon.
    /// Positive for counter-clockwise (exterior), negative for clockwise (hole).
    /// Uses the shoelace formula.
    pub fn signed_area(&self) -> CoordF {
        if self.points.len() < 3 {
            return 0.0;
        }

        let mut sum: i128 = 0;
        for i in 0..self.points.len() {
            let j = (i + 1) % self.points.len();
            sum += self.points[i].x as i128 * self.points[j].y as i128;
            sum -= self.points[j].x as i128 * self.points[i].y as i128;
        }

        sum as CoordF / 2.0
    }

    /// Calculate the unsigned area of the polygon.
    #[inline]
    pub fn area(&self) -> CoordF {
        self.signed_area().abs()
    }

    /// Check if the polygon is counter-clockwise (positive area).
    #[inline]
    pub fn is_counter_clockwise(&self) -> bool {
        self.signed_area() > 0.0
    }

    /// Check if the polygon is clockwise (negative area).
    #[inline]
    pub fn is_clockwise(&self) -> bool {
        self.signed_area() < 0.0
    }

    /// Ensure the polygon is counter-clockwise by reversing if necessary.
    pub fn make_counter_clockwise(&mut self) {
        if self.is_clockwise() {
            self.reverse();
        }
    }

    /// Ensure the polygon is clockwise by reversing if necessary.
    pub fn make_clockwise(&mut self) {
        if self.is_counter_clockwise() {
            self.reverse();
        }
    }

    /// Reverse the order of points in the polygon.
    pub fn reverse(&mut self) {
        self.points.reverse();
    }

    /// Return a reversed copy of the polygon.
    pub fn reversed(&self) -> Self {
        let mut result = self.clone();
        result.reverse();
        result
    }

    /// Calculate the perimeter (total edge length) of the polygon.
    pub fn perimeter(&self) -> CoordF {
        if self.points.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 0..self.points.len() {
            total += self.edge(i).length();
        }
        total
    }

    /// Calculate the centroid (center of mass) of the polygon.
    pub fn centroid(&self) -> Point {
        if self.points.is_empty() {
            return Point::zero();
        }

        if self.points.len() == 1 {
            return self.points[0];
        }

        if self.points.len() == 2 {
            return Point::new(
                (self.points[0].x + self.points[1].x) / 2,
                (self.points[0].y + self.points[1].y) / 2,
            );
        }

        let mut cx: i128 = 0;
        let mut cy: i128 = 0;
        let mut area: i128 = 0;

        for i in 0..self.points.len() {
            let j = (i + 1) % self.points.len();
            let cross = self.points[i].x as i128 * self.points[j].y as i128
                - self.points[j].x as i128 * self.points[i].y as i128;
            cx += (self.points[i].x as i128 + self.points[j].x as i128) * cross;
            cy += (self.points[i].y as i128 + self.points[j].y as i128) * cross;
            area += cross;
        }

        if area == 0 {
            // Degenerate polygon, return average of points
            let sum_x: i128 = self.points.iter().map(|p| p.x as i128).sum();
            let sum_y: i128 = self.points.iter().map(|p| p.y as i128).sum();
            return Point::new(
                (sum_x / self.points.len() as i128) as Coord,
                (sum_y / self.points.len() as i128) as Coord,
            );
        }

        Point::new((cx / (3 * area)) as Coord, (cy / (3 * area)) as Coord)
    }

    /// Get the bounding box of the polygon.
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox::from_points(&self.points)
    }

    /// Check if a point is inside the polygon using the ray casting algorithm.
    pub fn contains_point(&self, p: &Point) -> bool {
        if self.points.len() < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = self.points.len() - 1;

        for i in 0..self.points.len() {
            let pi = &self.points[i];
            let pj = &self.points[j];

            if ((pi.y > p.y) != (pj.y > p.y))
                && (p.x as i128)
                    < (pj.x as i128 - pi.x as i128) * (p.y as i128 - pi.y as i128)
                        / (pj.y as i128 - pi.y as i128)
                        + pi.x as i128
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    /// Check if a point is on the boundary of the polygon.
    pub fn is_point_on_boundary(&self, p: &Point, tolerance: Coord) -> bool {
        for edge in self.edges() {
            if edge.contains_point(p, tolerance) {
                return true;
            }
        }
        false
    }

    /// Find the closest point on the polygon boundary to the given point.
    pub fn closest_point(&self, p: &Point) -> Point {
        if self.points.is_empty() {
            return Point::zero();
        }

        if self.points.len() == 1 {
            return self.points[0];
        }

        let mut closest = self.points[0];
        let mut min_dist = i128::MAX;

        for edge in self.edges() {
            let proj = edge.project_point(p);
            let dist = p.distance_squared(&proj);
            if dist < min_dist {
                min_dist = dist;
                closest = proj;
            }
        }

        closest
    }

    /// Distance from a point to the polygon boundary.
    pub fn distance_to_point(&self, p: &Point) -> CoordF {
        let closest = self.closest_point(p);
        p.distance(&closest)
    }

    /// Translate the polygon by a vector.
    pub fn translate(&mut self, v: Point) {
        for p in &mut self.points {
            *p = *p + v;
        }
    }

    /// Return a translated copy of the polygon.
    pub fn translated(&self, v: Point) -> Self {
        let mut result = self.clone();
        result.translate(v);
        result
    }

    /// Scale the polygon about the origin.
    pub fn scale(&mut self, factor: CoordF) {
        for p in &mut self.points {
            *p = *p * factor;
        }
    }

    /// Return a scaled copy of the polygon.
    pub fn scaled(&self, factor: CoordF) -> Self {
        let mut result = self.clone();
        result.scale(factor);
        result
    }

    /// Rotate the polygon about the origin.
    pub fn rotate(&mut self, angle: CoordF) {
        for p in &mut self.points {
            *p = p.rotate(angle);
        }
    }

    /// Return a rotated copy of the polygon.
    pub fn rotated(&self, angle: CoordF) -> Self {
        let mut result = self.clone();
        result.rotate(angle);
        result
    }

    /// Rotate the polygon about a center point.
    pub fn rotate_around(&mut self, angle: CoordF, center: Point) {
        for p in &mut self.points {
            *p = p.rotate_around(angle, center);
        }
    }

    /// Return a copy rotated about a center point.
    pub fn rotated_around(&self, angle: CoordF, center: Point) -> Self {
        let mut result = self.clone();
        result.rotate_around(angle, center);
        result
    }

    /// Simplify the polygon by removing collinear and duplicate points.
    pub fn simplify(&mut self, tolerance: Coord) {
        if self.points.len() < 3 {
            return;
        }

        let mut new_points = Vec::with_capacity(self.points.len());
        let mut prev_idx = self.points.len() - 1;

        for i in 0..self.points.len() {
            let next_idx = (i + 1) % self.points.len();

            // Skip duplicate points
            if self.points[i].coincides_with(&self.points[next_idx], tolerance) {
                continue;
            }

            // Check if point is collinear with neighbors
            let prev = self.points[prev_idx];
            let curr = self.points[i];
            let next = self.points[next_idx];

            let line = Line::new(prev, next);
            let dist = line.distance_to_point(&curr);

            if dist > tolerance as CoordF {
                new_points.push(curr);
            }

            prev_idx = i;
        }

        self.points = new_points;
    }

    /// Return a simplified copy of the polygon.
    pub fn simplified(&self, tolerance: Coord) -> Self {
        let mut result = self.clone();
        result.simplify(tolerance);
        result
    }

    /// Check if this polygon is valid (has at least 3 non-collinear points).
    pub fn is_valid(&self) -> bool {
        if self.points.len() < 3 {
            return false;
        }

        // Check for non-zero area
        self.signed_area().abs() > 0.0
    }

    /// Convert to a polyline (open path).
    pub fn to_polyline(&self) -> Polyline {
        Polyline::from_points(self.points.clone())
    }

    /// Convert to a closed polyline (with first point repeated at end).
    pub fn to_closed_polyline(&self) -> Polyline {
        let mut points = self.points.clone();
        if !points.is_empty() {
            points.push(points[0]);
        }
        Polyline::from_points(points)
    }

    /// Split the polygon at a point, returning two polylines.
    /// The point should be on the polygon boundary.
    pub fn split_at_point(&self, p: Point) -> (Polyline, Polyline) {
        if self.points.is_empty() {
            return (Polyline::new(), Polyline::new());
        }

        // Find the edge containing the point
        let mut split_idx = 0;
        for i in 0..self.points.len() {
            let edge = self.edge(i);
            if edge.contains_point(&p, 1) {
                split_idx = i;
                break;
            }
        }

        let mut first = Vec::new();
        let mut second = Vec::new();

        first.push(p);
        for i in (split_idx + 1)..=split_idx + self.points.len() {
            first.push(self.point_at(i));
        }
        first.push(p);

        second.push(p);
        second.push(p);

        (Polyline::from_points(first), Polyline::from_points(second))
    }

    /// Create a rectangular polygon.
    pub fn rectangle(min: Point, max: Point) -> Self {
        Self::from_points(vec![
            min,
            Point::new(max.x, min.y),
            max,
            Point::new(min.x, max.y),
        ])
    }

    /// Create a square polygon centered at a point.
    pub fn square(center: Point, half_size: Coord) -> Self {
        Self::rectangle(
            Point::new(center.x - half_size, center.y - half_size),
            Point::new(center.x + half_size, center.y + half_size),
        )
    }

    /// Create a regular polygon with n sides, centered at origin.
    pub fn regular(n: usize, radius: Coord) -> Self {
        if n < 3 {
            return Self::new();
        }

        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * i as CoordF / n as CoordF;
            points.push(Point::new(
                (radius as CoordF * angle.cos()).round() as Coord,
                (radius as CoordF * angle.sin()).round() as Coord,
            ));
        }

        Self::from_points(points)
    }

    /// Create a circle approximation with n segments.
    pub fn circle(center: Point, radius: Coord, segments: usize) -> Self {
        let mut poly = Self::regular(segments, radius);
        poly.translate(center);
        poly
    }
}

impl fmt::Debug for Polygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Polygon({} points)", self.points.len())
    }
}

impl fmt::Display for Polygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Polygon[")?;
        for (i, p) in self.points.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", p)?;
        }
        write!(f, "]")
    }
}

impl Deref for Polygon {
    type Target = [Point];

    fn deref(&self) -> &Self::Target {
        &self.points
    }
}

impl DerefMut for Polygon {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.points
    }
}

impl Index<usize> for Polygon {
    type Output = Point;

    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl IndexMut<usize> for Polygon {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.points[index]
    }
}

impl FromIterator<Point> for Polygon {
    fn from_iter<I: IntoIterator<Item = Point>>(iter: I) -> Self {
        Self {
            points: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Polygon {
    type Item = Point;
    type IntoIter = std::vec::IntoIter<Point>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter()
    }
}

impl<'a> IntoIterator for &'a Polygon {
    type Item = &'a Point;
    type IntoIter = std::slice::Iter<'a, Point>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

impl<'a> IntoIterator for &'a mut Polygon {
    type Item = &'a mut Point;
    type IntoIter = std::slice::IterMut<'a, Point>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.iter_mut()
    }
}

impl From<Vec<Point>> for Polygon {
    fn from(points: Vec<Point>) -> Self {
        Self::from_points(points)
    }
}

impl From<Polygon> for Vec<Point> {
    fn from(polygon: Polygon) -> Self {
        polygon.into_points()
    }
}

/// Type alias for a collection of polygons.
pub type Polygons = Vec<Polygon>;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square() -> Polygon {
        Polygon::from_points(vec![
            Point::new(0, 0),
            Point::new(100, 0),
            Point::new(100, 100),
            Point::new(0, 100),
        ])
    }

    #[test]
    fn test_polygon_new() {
        let poly = Polygon::new();
        assert!(poly.is_empty());
        assert_eq!(poly.len(), 0);
    }

    #[test]
    fn test_polygon_from_points() {
        let poly = make_square();
        assert_eq!(poly.len(), 4);
        assert!(!poly.is_empty());
    }

    #[test]
    fn test_polygon_edge() {
        let poly = make_square();
        let edge = poly.edge(0);
        assert_eq!(edge.a, Point::new(0, 0));
        assert_eq!(edge.b, Point::new(100, 0));

        // Test wrap-around
        let last_edge = poly.edge(3);
        assert_eq!(last_edge.a, Point::new(0, 100));
        assert_eq!(last_edge.b, Point::new(0, 0));
    }

    #[test]
    fn test_polygon_edges() {
        let poly = make_square();
        let edges = poly.edges();
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn test_polygon_area() {
        let poly = make_square();
        let area = poly.area();
        assert!((area - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_polygon_signed_area() {
        let ccw = make_square();
        assert!(ccw.signed_area() > 0.0);

        let cw = ccw.reversed();
        assert!(cw.signed_area() < 0.0);
    }

    #[test]
    fn test_polygon_is_counter_clockwise() {
        let poly = make_square();
        assert!(poly.is_counter_clockwise());
        assert!(!poly.is_clockwise());
    }

    #[test]
    fn test_polygon_perimeter() {
        let poly = make_square();
        let perim = poly.perimeter();
        assert!((perim - 400.0).abs() < 1.0);
    }

    #[test]
    fn test_polygon_centroid() {
        let poly = make_square();
        let centroid = poly.centroid();
        assert_eq!(centroid.x, 50);
        assert_eq!(centroid.y, 50);
    }

    #[test]
    fn test_polygon_bounding_box() {
        let poly = make_square();
        let bb = poly.bounding_box();
        assert_eq!(bb.min.x, 0);
        assert_eq!(bb.min.y, 0);
        assert_eq!(bb.max.x, 100);
        assert_eq!(bb.max.y, 100);
    }

    #[test]
    fn test_polygon_contains_point() {
        let poly = make_square();

        // Point inside
        assert!(poly.contains_point(&Point::new(50, 50)));

        // Points outside
        assert!(!poly.contains_point(&Point::new(-10, 50)));
        assert!(!poly.contains_point(&Point::new(110, 50)));
        assert!(!poly.contains_point(&Point::new(50, -10)));
        assert!(!poly.contains_point(&Point::new(50, 110)));
    }

    #[test]
    fn test_polygon_translate() {
        let mut poly = make_square();
        poly.translate(Point::new(10, 20));

        assert_eq!(poly[0], Point::new(10, 20));
        assert_eq!(poly[1], Point::new(110, 20));
    }

    #[test]
    fn test_polygon_scale() {
        let mut poly = make_square();
        poly.scale(2.0);

        assert_eq!(poly[0], Point::new(0, 0));
        assert_eq!(poly[2], Point::new(200, 200));
    }

    #[test]
    fn test_polygon_reverse() {
        let mut poly = make_square();
        let original_first = poly[0];
        let original_last = poly[3];

        poly.reverse();

        assert_eq!(poly[0], original_last);
        assert_eq!(poly[3], original_first);
    }

    #[test]
    fn test_polygon_rectangle() {
        let poly = Polygon::rectangle(Point::new(0, 0), Point::new(100, 50));
        assert_eq!(poly.len(), 4);
        assert!((poly.area() - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_polygon_regular() {
        let triangle = Polygon::regular(3, 100);
        assert_eq!(triangle.len(), 3);

        let hexagon = Polygon::regular(6, 100);
        assert_eq!(hexagon.len(), 6);
    }

    #[test]
    fn test_polygon_is_valid() {
        let poly = make_square();
        assert!(poly.is_valid());

        // A line is not valid
        let line = Polygon::from_points(vec![Point::new(0, 0), Point::new(100, 0)]);
        assert!(!line.is_valid());

        // Empty polygon is not valid
        let empty = Polygon::new();
        assert!(!empty.is_valid());
    }

    #[test]
    fn test_polygon_closest_point() {
        let poly = make_square();
        let p = Point::new(50, -20);
        let closest = poly.closest_point(&p);
        assert_eq!(closest.x, 50);
        assert_eq!(closest.y, 0);
    }

    #[test]
    fn test_polygon_iterator() {
        let poly = make_square();
        let mut count = 0;
        for _ in &poly {
            count += 1;
        }
        assert_eq!(count, 4);
    }
}
