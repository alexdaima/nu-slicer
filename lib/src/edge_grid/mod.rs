//! EdgeGrid - Spatial acceleration structure for polygon edge queries.
//!
//! This module provides a grid-based spatial index for efficient queries on polygon edges,
//! including intersection testing, closest point queries, and signed distance calculations.
//!
//! # libslic3r Mapping
//!
//! This module corresponds to `EdgeGrid.cpp` and `EdgeGrid.hpp` in BambuStudio/libslic3r.
//!
//! # Key Features
//!
//! - Fast line-polygon intersection testing
//! - Closest point on polygon edge queries
//! - Signed distance field computation
//! - Support for both open polylines and closed polygons
//!
//! # Example
//!
//! ```ignore
//! use slicer::edge_grid::EdgeGrid;
//! use slicer::geometry::{Polygon, Point};
//!
//! let polygons = vec![Polygon::from_points(vec![
//!     Point::new(0, 0),
//!     Point::new(1000000, 0),
//!     Point::new(1000000, 1000000),
//!     Point::new(0, 1000000),
//! ])];
//!
//! let grid = EdgeGrid::from_polygons(&polygons, 100000); // 0.1mm resolution
//!
//! // Check if a line intersects any polygon edge
//! let intersects = grid.line_intersects_any(&Point::new(500000, -100000), &Point::new(500000, 500000));
//! ```

use crate::geometry::{BoundingBox, Line, Point, Polygon, Polyline};

/// A contour represents a sequence of points forming either an open polyline or closed polygon.
#[derive(Clone, Debug)]
pub struct Contour {
    /// Points of the contour
    points: Vec<Point>,
    /// Whether this contour is open (polyline) or closed (polygon)
    open: bool,
}

impl Contour {
    /// Create a new closed contour from points.
    pub fn new_closed(points: Vec<Point>) -> Self {
        Self {
            points,
            open: false,
        }
    }

    /// Create a new open contour from points.
    pub fn new_open(points: Vec<Point>) -> Self {
        Self { points, open: true }
    }

    /// Create from a polygon (closed).
    pub fn from_polygon(polygon: &Polygon) -> Self {
        Self::new_closed(polygon.points().to_vec())
    }

    /// Create from a polyline (open).
    pub fn from_polyline(polyline: &Polyline) -> Self {
        Self::new_open(polyline.points().to_vec())
    }

    /// Returns true if this contour is open (polyline).
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Returns true if this contour is closed (polygon).
    pub fn is_closed(&self) -> bool {
        !self.open
    }

    /// Get the points of this contour.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Get the number of segments in this contour.
    pub fn num_segments(&self) -> usize {
        if self.points.len() < 2 {
            return 0;
        }
        if self.open {
            self.points.len() - 1
        } else {
            self.points.len()
        }
    }

    /// Get the start point of a segment.
    pub fn segment_start(&self, idx: usize) -> &Point {
        &self.points[idx]
    }

    /// Get the end point of a segment.
    pub fn segment_end(&self, idx: usize) -> &Point {
        let next_idx = if idx + 1 >= self.points.len() {
            0
        } else {
            idx + 1
        };
        &self.points[next_idx]
    }

    /// Get a segment as a Line.
    pub fn segment(&self, idx: usize) -> Line {
        Line::new(*self.segment_start(idx), *self.segment_end(idx))
    }

    /// Get all segments as Lines.
    pub fn segments(&self) -> Vec<Line> {
        (0..self.num_segments()).map(|i| self.segment(i)).collect()
    }
}

/// A cell in the edge grid.
#[derive(Clone, Debug, Default)]
struct Cell {
    /// Start index in the cell_data array.
    begin: usize,
    /// End index in the cell_data array (exclusive).
    end: usize,
}

impl Cell {
    fn is_empty(&self) -> bool {
        self.begin >= self.end
    }
}

/// Result of a closest point query.
#[derive(Clone, Debug)]
pub struct ClosestPointResult {
    /// Index of the contour.
    pub contour_idx: usize,
    /// Index of the segment start point.
    pub start_point_idx: usize,
    /// Signed distance to the closest point.
    pub distance: f64,
    /// Parameter t on the segment [0, 1).
    pub t: f64,
    /// The closest point itself.
    pub point: Point,
}

impl ClosestPointResult {
    /// Create an invalid result.
    pub fn invalid() -> Self {
        Self {
            contour_idx: usize::MAX,
            start_point_idx: usize::MAX,
            distance: f64::MAX,
            t: 0.0,
            point: Point::new(0, 0),
        }
    }

    /// Check if this result is valid.
    pub fn is_valid(&self) -> bool {
        self.contour_idx != usize::MAX
    }
}

/// Intersection result.
#[derive(Clone, Debug)]
pub struct Intersection {
    /// Index of the contour (boundary polygon).
    pub contour_idx: usize,
    /// Index of the segment within the contour.
    pub segment_idx: usize,
    /// The intersection point.
    pub point: Point,
    /// Distance along the original line from start.
    pub distance: f64,
}

/// EdgeGrid - A spatial acceleration structure for polygon edges.
///
/// The grid divides the bounding box into cells and stores which polygon edges
/// pass through each cell, enabling fast spatial queries.
#[derive(Clone, Debug)]
pub struct EdgeGrid {
    /// Bounding box of the grid.
    bbox: BoundingBox,
    /// Resolution (cell size) in scaled coordinates.
    resolution: i64,
    /// Number of rows in the grid.
    rows: usize,
    /// Number of columns in the grid.
    cols: usize,
    /// Contours stored in the grid.
    contours: Vec<Contour>,
    /// Cell data: (contour_idx, segment_idx) pairs.
    cell_data: Vec<(usize, usize)>,
    /// Cells indexing into cell_data.
    cells: Vec<Cell>,
    /// Pre-computed signed distance field (optional).
    signed_distance_field: Vec<f32>,
}

impl EdgeGrid {
    /// Create a new empty EdgeGrid.
    pub fn new() -> Self {
        Self {
            bbox: BoundingBox::new(),
            resolution: 1,
            rows: 0,
            cols: 0,
            contours: Vec::new(),
            cell_data: Vec::new(),
            cells: Vec::new(),
            signed_distance_field: Vec::new(),
        }
    }

    /// Create an EdgeGrid from polygons with the given resolution.
    pub fn from_polygons(polygons: &[Polygon], resolution: i64) -> Self {
        let mut grid = Self::new();
        grid.create_from_polygons(polygons, resolution);
        grid
    }

    /// Create an EdgeGrid from a single polygon.
    pub fn from_polygon(polygon: &Polygon, resolution: i64) -> Self {
        Self::from_polygons(&[polygon.clone()], resolution)
    }

    /// Set the bounding box.
    pub fn set_bbox(&mut self, bbox: BoundingBox) {
        self.bbox = bbox;
    }

    /// Get the bounding box.
    pub fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// Get the resolution.
    pub fn resolution(&self) -> i64 {
        self.resolution
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the contours.
    pub fn contours(&self) -> &[Contour] {
        &self.contours
    }

    /// Create the grid from polygons.
    pub fn create_from_polygons(&mut self, polygons: &[Polygon], resolution: i64) {
        self.contours = polygons.iter().map(Contour::from_polygon).collect();
        self.create_from_contours(resolution);
    }

    /// Create the grid from polylines.
    pub fn create_from_polylines(&mut self, polylines: &[Polyline], resolution: i64) {
        self.contours = polylines.iter().map(Contour::from_polyline).collect();
        self.create_from_contours(resolution);
    }

    /// Create the grid from both polygons and polylines.
    pub fn create_from_mixed(
        &mut self,
        polygons: &[Polygon],
        polylines: &[Polyline],
        resolution: i64,
    ) {
        self.contours = polygons
            .iter()
            .map(Contour::from_polygon)
            .chain(polylines.iter().map(Contour::from_polyline))
            .collect();
        self.create_from_contours(resolution);
    }

    /// Internal: create the grid from stored contours.
    fn create_from_contours(&mut self, resolution: i64) {
        self.resolution = resolution.max(1);

        // Compute bounding box
        self.bbox = BoundingBox::new();
        for contour in &self.contours {
            for point in contour.points() {
                self.bbox.merge_point(*point);
            }
        }

        if self.bbox.is_empty() {
            self.rows = 0;
            self.cols = 0;
            self.cells.clear();
            self.cell_data.clear();
            return;
        }

        // Add a small margin to avoid edge cases
        let margin = self.resolution;
        self.bbox = BoundingBox::from_points_minmax(
            Point::new(self.bbox.min.x - margin, self.bbox.min.y - margin),
            Point::new(self.bbox.max.x + margin, self.bbox.max.y + margin),
        );

        // Calculate grid dimensions
        let size = self.bbox.size();
        self.cols = ((size.x as i64 + self.resolution - 1) / self.resolution).max(1) as usize;
        self.rows = ((size.y as i64 + self.resolution - 1) / self.resolution).max(1) as usize;

        // Count edges per cell
        let num_cells = self.rows * self.cols;
        let mut cell_counts = vec![0usize; num_cells];

        for (contour_idx, contour) in self.contours.iter().enumerate() {
            for seg_idx in 0..contour.num_segments() {
                let p1 = contour.segment_start(seg_idx);
                let p2 = contour.segment_end(seg_idx);
                self.visit_cells_for_segment(p1, p2, |row, col| {
                    let cell_idx = row * self.cols + col;
                    if cell_idx < num_cells {
                        cell_counts[cell_idx] += 1;
                    }
                });
                // Mark we processed this segment (for borrow checker)
                let _ = (contour_idx, seg_idx);
            }
        }

        // Build cell offsets
        self.cells = vec![Cell::default(); num_cells];
        let mut offset = 0;
        for (i, count) in cell_counts.iter().enumerate() {
            self.cells[i].begin = offset;
            self.cells[i].end = offset;
            offset += count;
        }

        // Allocate cell data
        self.cell_data = vec![(0, 0); offset];

        // Collect segment data first to avoid borrow checker issues
        let segment_data: Vec<(usize, usize, Point, Point)> = self
            .contours
            .iter()
            .enumerate()
            .flat_map(|(contour_idx, contour)| {
                (0..contour.num_segments()).map(move |seg_idx| {
                    let p1 = *contour.segment_start(seg_idx);
                    let p2 = *contour.segment_end(seg_idx);
                    (contour_idx, seg_idx, p1, p2)
                })
            })
            .collect();

        // Fill cell data
        for (contour_idx, seg_idx, p1, p2) in segment_data {
            self.visit_cells_for_segment_mut(&p1, &p2, contour_idx, seg_idx);
        }
    }

    /// Convert a point to cell coordinates.
    fn point_to_cell(&self, point: &Point) -> (usize, usize) {
        let x = ((point.x - self.bbox.min.x) / self.resolution).max(0) as usize;
        let y = ((point.y - self.bbox.min.y) / self.resolution).max(0) as usize;
        (
            y.min(self.rows.saturating_sub(1)),
            x.min(self.cols.saturating_sub(1)),
        )
    }

    /// Visit all cells that a line segment passes through.
    fn visit_cells_for_segment<F>(&self, p1: &Point, p2: &Point, mut visitor: F)
    where
        F: FnMut(usize, usize),
    {
        if self.cols == 0 || self.rows == 0 {
            return;
        }

        let (row1, col1) = self.point_to_cell(p1);
        let (row2, col2) = self.point_to_cell(p2);

        // Use Bresenham-like algorithm for cell traversal
        let col1 = col1 as i64;
        let row1 = row1 as i64;
        let col2 = col2 as i64;
        let row2 = row2 as i64;

        let dx = (col2 - col1).abs();
        let dy = (row2 - row1).abs();
        let sx: i64 = if col1 < col2 { 1 } else { -1 };
        let sy: i64 = if row1 < row2 { 1 } else { -1 };

        let mut col = col1;
        let mut row = row1;
        let mut err = dx - dy;

        loop {
            if col >= 0 && col < self.cols as i64 && row >= 0 && row < self.rows as i64 {
                visitor(row as usize, col as usize);
            }

            if col == col2 && row == row2 {
                break;
            }

            let e2 = 2 * err;

            // Move diagonally or along one axis
            if e2 > -dy && e2 < dx {
                // Move diagonally
                err += -dy + dx;
                col += sx;
                row += sy;
            } else if e2 > -dy {
                err -= dy;
                col += sx;
            } else {
                err += dx;
                row += sy;
            }
        }
    }

    /// Visit cells and insert segment into cell_data.
    fn visit_cells_for_segment_mut(
        &mut self,
        p1: &Point,
        p2: &Point,
        contour_idx: usize,
        seg_idx: usize,
    ) {
        if self.cols == 0 || self.rows == 0 {
            return;
        }

        let (row1, col1) = self.point_to_cell(p1);
        let (row2, col2) = self.point_to_cell(p2);

        let col1 = col1 as i64;
        let row1 = row1 as i64;
        let col2 = col2 as i64;
        let row2 = row2 as i64;

        let dx = (col2 - col1).abs();
        let dy = (row2 - row1).abs();
        let sx: i64 = if col1 < col2 { 1 } else { -1 };
        let sy: i64 = if row1 < row2 { 1 } else { -1 };

        let mut col = col1;
        let mut row = row1;
        let mut err = dx - dy;

        loop {
            if col >= 0 && col < self.cols as i64 && row >= 0 && row < self.rows as i64 {
                let cell_idx = row as usize * self.cols + col as usize;
                if cell_idx < self.cells.len() {
                    let insert_idx = self.cells[cell_idx].end;
                    if insert_idx < self.cell_data.len() {
                        self.cell_data[insert_idx] = (contour_idx, seg_idx);
                        self.cells[cell_idx].end += 1;
                    }
                }
            }

            if col == col2 && row == row2 {
                break;
            }

            let e2 = 2 * err;

            if e2 > -dy && e2 < dx {
                err += -dy + dx;
                col += sx;
                row += sy;
            } else if e2 > -dy {
                err -= dy;
                col += sx;
            } else {
                err += dx;
                row += sy;
            }
        }
    }

    /// Get the cell data range for a cell at (row, col).
    fn cell_data_range(&self, row: usize, col: usize) -> &[(usize, usize)] {
        if row >= self.rows || col >= self.cols {
            return &[];
        }
        let cell_idx = row * self.cols + col;
        if cell_idx >= self.cells.len() {
            return &[];
        }
        let cell = &self.cells[cell_idx];
        if cell.begin >= cell.end || cell.end > self.cell_data.len() {
            return &[];
        }
        &self.cell_data[cell.begin..cell.end]
    }

    /// Check if a line intersects any edge in the grid.
    pub fn line_intersects_any(&self, p1: &Point, p2: &Point) -> bool {
        if self.cols == 0 || self.rows == 0 {
            return false;
        }

        let line = Line::new(*p1, *p2);
        let mut found = false;

        self.visit_cells_for_segment(p1, p2, |row, col| {
            if found {
                return;
            }
            for &(contour_idx, seg_idx) in self.cell_data_range(row, col) {
                let segment = self.contours[contour_idx].segment(seg_idx);
                if line.intersects(&segment) {
                    found = true;
                    return;
                }
            }
        });

        found
    }

    /// Find all intersections between a line and edges in the grid.
    pub fn find_intersections(&self, p1: &Point, p2: &Point) -> Vec<Intersection> {
        let mut intersections = Vec::new();
        let mut seen = std::collections::HashSet::new();

        if self.cols == 0 || self.rows == 0 {
            return intersections;
        }

        let line = Line::new(*p1, *p2);
        let line_vec = (p2.x - p1.x, p2.y - p1.y);

        self.visit_cells_for_segment(p1, p2, |row, col| {
            for &(contour_idx, seg_idx) in self.cell_data_range(row, col) {
                // Avoid duplicates
                if !seen.insert((contour_idx, seg_idx)) {
                    continue;
                }

                let segment = self.contours[contour_idx].segment(seg_idx);
                if let Some(intersection_point) = line.intersection(&segment) {
                    // Calculate distance along line
                    let dx = intersection_point.x - p1.x;
                    let dy = intersection_point.y - p1.y;
                    let distance = if line_vec.0.abs() > line_vec.1.abs() {
                        dx as f64 / line_vec.0 as f64
                    } else if line_vec.1 != 0 {
                        dy as f64 / line_vec.1 as f64
                    } else {
                        0.0
                    };

                    intersections.push(Intersection {
                        contour_idx,
                        segment_idx: seg_idx,
                        point: intersection_point,
                        distance,
                    });
                }
            }
        });

        // Sort by distance along the line
        intersections.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        intersections
    }

    /// Find the closest point on any edge to a query point within a search radius.
    pub fn closest_point(&self, query: &Point, search_radius: i64) -> ClosestPointResult {
        let mut result = ClosestPointResult::invalid();

        if self.cols == 0 || self.rows == 0 {
            return result;
        }

        let search_radius_sq = (search_radius as f64) * (search_radius as f64);

        // Determine cell range to search
        let min_cell = self.point_to_cell(&Point::new(
            query.x - search_radius,
            query.y - search_radius,
        ));
        let max_cell = self.point_to_cell(&Point::new(
            query.x + search_radius,
            query.y + search_radius,
        ));

        let mut seen = std::collections::HashSet::new();
        let query_f = (query.x as f64, query.y as f64);

        for row in min_cell.0..=max_cell.0 {
            for col in min_cell.1..=max_cell.1 {
                for &(contour_idx, seg_idx) in self.cell_data_range(row, col) {
                    if !seen.insert((contour_idx, seg_idx)) {
                        continue;
                    }

                    let contour = &self.contours[contour_idx];
                    let p1 = contour.segment_start(seg_idx);
                    let p2 = contour.segment_end(seg_idx);

                    // Find closest point on segment
                    let (closest, t) = closest_point_on_segment(query_f, p1, p2);
                    let dx = closest.0 - query_f.0;
                    let dy = closest.1 - query_f.1;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < search_radius_sq && dist_sq < result.distance * result.distance {
                        result.contour_idx = contour_idx;
                        result.start_point_idx = seg_idx;
                        result.distance = dist_sq.sqrt();
                        result.t = t;
                        result.point = Point::new(closest.0 as i64, closest.1 as i64);
                    }
                }
            }
        }

        result
    }

    /// Check if a point is inside the polygon (for closed contours).
    /// Uses ray casting algorithm.
    pub fn point_inside(&self, point: &Point) -> bool {
        if self.contours.is_empty() {
            return false;
        }

        let mut crossings = 0;

        for contour in &self.contours {
            if contour.is_open() {
                continue;
            }

            for seg_idx in 0..contour.num_segments() {
                let p1 = contour.segment_start(seg_idx);
                let p2 = contour.segment_end(seg_idx);

                // Check if horizontal ray from point crosses this segment
                if (p1.y > point.y) != (p2.y > point.y) {
                    let slope = (p2.x - p1.x) as f64 / (p2.y - p1.y) as f64;
                    let x_intersect = p1.x as f64 + (point.y - p1.y) as f64 * slope;
                    if (point.x as f64) < x_intersect {
                        crossings += 1;
                    }
                }
            }
        }

        crossings % 2 == 1
    }

    /// Calculate the signed distance field for all grid cells.
    pub fn calculate_sdf(&mut self) {
        let num_cells = self.rows * self.cols;
        self.signed_distance_field = vec![f32::MAX; num_cells];

        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell_idx = row * self.cols + col;
                let center = Point::new(
                    self.bbox.min.x + (col as i64) * self.resolution + self.resolution / 2,
                    self.bbox.min.y + (row as i64) * self.resolution + self.resolution / 2,
                );

                // Find closest edge
                let result = self.closest_point(&center, self.resolution * 10);
                let dist = if result.is_valid() {
                    result.distance as f32
                } else {
                    f32::MAX
                };

                // Determine sign based on inside/outside
                let sign = if self.point_inside(&center) {
                    -1.0
                } else {
                    1.0
                };

                self.signed_distance_field[cell_idx] = sign * dist;
            }
        }
    }

    /// Get the signed distance at a point using bilinear interpolation.
    pub fn signed_distance_bilinear(&self, point: &Point) -> f32 {
        if self.signed_distance_field.is_empty() {
            return f32::MAX;
        }

        let fx = (point.x - self.bbox.min.x) as f64 / self.resolution as f64;
        let fy = (point.y - self.bbox.min.y) as f64 / self.resolution as f64;

        let x0 = fx.floor() as i64;
        let y0 = fy.floor() as i64;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;

        let get_sdf = |row: i64, col: i64| -> f32 {
            if row < 0 || col < 0 || row >= self.rows as i64 || col >= self.cols as i64 {
                return f32::MAX;
            }
            self.signed_distance_field[row as usize * self.cols + col as usize]
        };

        let v00 = get_sdf(y0, x0);
        let v10 = get_sdf(y0, x1);
        let v01 = get_sdf(y1, x0);
        let v11 = get_sdf(y1, x1);

        let v0 = v00 * (1.0 - tx as f32) + v10 * tx as f32;
        let v1 = v01 * (1.0 - tx as f32) + v11 * tx as f32;

        v0 * (1.0 - ty as f32) + v1 * ty as f32
    }
}

impl Default for EdgeGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the closest point on a line segment to a query point.
/// Returns (closest_point, t) where t is the parameter along the segment [0, 1].
fn closest_point_on_segment(query: (f64, f64), p1: &Point, p2: &Point) -> ((f64, f64), f64) {
    let p1f = (p1.x as f64, p1.y as f64);
    let p2f = (p2.x as f64, p2.y as f64);

    let dx = p2f.0 - p1f.0;
    let dy = p2f.1 - p1f.1;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        // Degenerate segment
        return (p1f, 0.0);
    }

    // Project query onto segment
    let t = ((query.0 - p1f.0) * dx + (query.1 - p1f.1) * dy) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);

    let closest = (p1f.0 + t_clamped * dx, p1f.1 + t_clamped * dy);

    (closest, t_clamped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square() -> Polygon {
        Polygon::from_points(vec![
            Point::new(0, 0),
            Point::new(1_000_000, 0),
            Point::new(1_000_000, 1_000_000),
            Point::new(0, 1_000_000),
        ])
    }

    #[test]
    fn test_contour_from_polygon() {
        let square = make_square();
        let contour = Contour::from_polygon(&square);

        assert!(contour.is_closed());
        assert!(!contour.is_open());
        assert_eq!(contour.num_segments(), 4);
    }

    #[test]
    fn test_contour_segments() {
        let square = make_square();
        let contour = Contour::from_polygon(&square);

        let seg0 = contour.segment(0);
        assert_eq!(seg0.a, Point::new(0, 0));
        assert_eq!(seg0.b, Point::new(1_000_000, 0));

        let seg3 = contour.segment(3);
        assert_eq!(seg3.a, Point::new(0, 1_000_000));
        assert_eq!(seg3.b, Point::new(0, 0));
    }

    #[test]
    fn test_edge_grid_creation() {
        let square = make_square();
        let grid = EdgeGrid::from_polygon(&square, 100_000);

        assert!(!grid.bbox().is_empty());
        assert!(grid.rows() > 0);
        assert!(grid.cols() > 0);
        assert_eq!(grid.contours().len(), 1);
    }

    #[test]
    fn test_line_intersects_any() {
        let square = make_square();
        let grid = EdgeGrid::from_polygon(&square, 100_000);

        // Line from outside to inside should intersect
        let p1 = Point::new(-500_000, 500_000);
        let p2 = Point::new(500_000, 500_000);
        assert!(grid.line_intersects_any(&p1, &p2));

        // Line completely outside should not intersect
        let p1 = Point::new(-500_000, -500_000);
        let p2 = Point::new(-100_000, -100_000);
        assert!(!grid.line_intersects_any(&p1, &p2));
    }

    #[test]
    fn test_find_intersections() {
        let square = make_square();
        let grid = EdgeGrid::from_polygon(&square, 100_000);

        // Line that crosses two edges
        let p1 = Point::new(-500_000, 500_000);
        let p2 = Point::new(1_500_000, 500_000);
        let intersections = grid.find_intersections(&p1, &p2);

        assert_eq!(intersections.len(), 2);
    }

    #[test]
    fn test_closest_point() {
        let square = make_square();
        let grid = EdgeGrid::from_polygon(&square, 100_000);

        // Point outside, closest to bottom edge
        let query = Point::new(500_000, -100_000);
        let result = grid.closest_point(&query, 200_000);

        assert!(result.is_valid());
        assert_eq!(result.point.y, 0); // Should be on bottom edge
        assert!((result.distance - 100_000.0).abs() < 1000.0);
    }

    #[test]
    fn test_point_inside() {
        let square = make_square();
        let grid = EdgeGrid::from_polygon(&square, 100_000);

        // Point inside
        assert!(grid.point_inside(&Point::new(500_000, 500_000)));

        // Point outside
        assert!(!grid.point_inside(&Point::new(-100_000, 500_000)));
    }

    #[test]
    fn test_closest_point_on_segment() {
        let p1 = Point::new(0, 0);
        let p2 = Point::new(1_000_000, 0);

        // Query point perpendicular to segment midpoint
        let query = (500_000.0, 100_000.0);
        let (closest, t) = closest_point_on_segment(query, &p1, &p2);

        assert!((closest.0 - 500_000.0).abs() < 1.0);
        assert!((closest.1 - 0.0).abs() < 1.0);
        assert!((t - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_grid() {
        let grid = EdgeGrid::new();

        assert!(!grid.line_intersects_any(&Point::new(0, 0), &Point::new(1, 1)));
        assert!(grid
            .find_intersections(&Point::new(0, 0), &Point::new(1, 1))
            .is_empty());
        assert!(!grid.closest_point(&Point::new(0, 0), 1000).is_valid());
    }

    #[test]
    fn test_multiple_polygons() {
        let square1 = Polygon::from_points(vec![
            Point::new(0, 0),
            Point::new(100_000, 0),
            Point::new(100_000, 100_000),
            Point::new(0, 100_000),
        ]);
        let square2 = Polygon::from_points(vec![
            Point::new(200_000, 0),
            Point::new(300_000, 0),
            Point::new(300_000, 100_000),
            Point::new(200_000, 100_000),
        ]);

        let grid = EdgeGrid::from_polygons(&[square1, square2], 10_000);

        assert_eq!(grid.contours().len(), 2);

        // Line through both squares
        let p1 = Point::new(-50_000, 50_000);
        let p2 = Point::new(350_000, 50_000);
        let intersections = grid.find_intersections(&p1, &p2);

        // Should intersect 4 edges (2 per square)
        assert_eq!(intersections.len(), 4);
    }

    #[test]
    fn test_contour_open_polyline() {
        let polyline = Polyline::from_points(vec![
            Point::new(0, 0),
            Point::new(100_000, 0),
            Point::new(100_000, 100_000),
        ]);
        let contour = Contour::from_polyline(&polyline);

        assert!(contour.is_open());
        assert_eq!(contour.num_segments(), 2); // Open: n-1 segments
    }
}
