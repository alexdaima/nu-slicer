//! Adaptive cubic infill implementation.
//!
//! This module provides adaptive infill that automatically varies density based on
//! proximity to the model surface. Areas near surfaces get denser infill for better
//! support, while interior regions use sparser infill for material/time savings.
//!
//! # Algorithm Overview
//!
//! 1. Build an octree from the mesh triangles
//! 2. Subdivide octree cells that contain triangles (recursive subdivision)
//! 3. At each layer Z height, extract infill lines from octree cells
//! 4. Lines are generated in 3 directions (rotated cube orientation)
//! 5. Connect lines using hooks for continuous extrusion
//! 6. Clip to the infill boundary
//!
//! # BambuStudio Reference
//!
//! This corresponds to:
//! - `src/libslic3r/Fill/FillAdaptive.cpp`
//! - `src/libslic3r/Fill/FillAdaptive.hpp`
//!
//! The algorithm is inspired by Cura's adaptive cubic infill:
//! - https://github.com/Ultimaker/CuraEngine/issues/381
//! - https://github.com/Ultimaker/CuraEngine/pull/401

use crate::geometry::{ExPolygon, Point, Polyline};
use crate::{scale, Coord, CoordF};
use std::f64::consts::PI;

/// Configuration for adaptive infill generation.
#[derive(Debug, Clone)]
pub struct AdaptiveInfillConfig {
    /// Line spacing (distance between infill lines in mm).
    pub line_spacing: CoordF,

    /// Extrusion width for infill lines (mm).
    pub extrusion_width: CoordF,

    /// Whether to only densify below internal overhangs.
    pub support_overhangs_only: bool,

    /// Hook length for connecting lines (mm).
    pub hook_length: CoordF,

    /// Maximum hook length (mm).
    pub hook_length_max: CoordF,

    /// Whether to connect infill lines.
    pub connect_lines: bool,
}

impl Default for AdaptiveInfillConfig {
    fn default() -> Self {
        Self {
            line_spacing: 2.0,
            extrusion_width: 0.45,
            support_overhangs_only: false,
            hook_length: 1.0,
            hook_length_max: 2.0,
            connect_lines: true,
        }
    }
}

impl AdaptiveInfillConfig {
    /// Create config from infill density (0.0 - 1.0).
    pub fn from_density(density: CoordF, extrusion_width: CoordF) -> Self {
        let density = density.clamp(0.01, 1.0);
        Self {
            line_spacing: extrusion_width / density,
            extrusion_width,
            ..Default::default()
        }
    }
}

/// Properties for cubes at each level of the octree.
#[derive(Debug, Clone)]
pub struct CubeProperties {
    /// Edge length of the cube.
    pub edge_length: CoordF,

    /// Height of the rotated cube (standing on corner).
    pub height: CoordF,

    /// Length of diagonal across a cube face.
    pub diagonal_length: CoordF,

    /// Max Z distance from cube center to generate lines.
    pub line_z_distance: CoordF,

    /// Max XY distance from cube center to generate lines.
    pub line_xy_distance: CoordF,
}

impl CubeProperties {
    /// Create cube properties for a given edge length.
    pub fn new(edge_length: CoordF) -> Self {
        Self {
            edge_length,
            height: edge_length * 3.0_f64.sqrt(),
            diagonal_length: edge_length * 2.0_f64.sqrt(),
            line_z_distance: edge_length / 3.0_f64.sqrt(),
            line_xy_distance: edge_length / 6.0_f64.sqrt(),
        }
    }
}

/// Generate cube properties for all octree levels.
fn make_cubes_properties(
    max_cube_edge_length: CoordF,
    line_spacing: CoordF,
) -> Vec<CubeProperties> {
    let max_edge = max_cube_edge_length + 1e-6;
    let mut properties = Vec::new();

    let mut edge_length = line_spacing * 2.0;
    loop {
        properties.push(CubeProperties::new(edge_length));
        if edge_length > max_edge {
            break;
        }
        edge_length *= 2.0;
    }

    properties
}

/// Ordering of children cubes in the octree.
const CHILD_CENTERS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

/// Traversal order of octree children cells for three infill directions.
/// This ensures lines are discretized in strictly monotonic order.
const CHILD_TRAVERSAL_ORDER: [[usize; 8]; 3] = [
    [2, 3, 0, 1, 6, 7, 4, 5],
    [4, 0, 6, 2, 5, 1, 7, 3],
    [1, 5, 0, 4, 3, 7, 2, 6],
];

/// Rotation angles for the octree coordinate system.
/// The octree is rotated so it stands on one of its corners.
const OCTREE_ROT: [f64; 3] = [
    5.0 * PI / 4.0,           // X rotation
    215.264_f64.to_radians(), // Y rotation (arctan(1/sqrt(2)) + 180°)
    PI / 6.0,                 // Z rotation
];

/// 3D vector type for octree calculations.
#[derive(Debug, Clone, Copy)]
pub struct Vec3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3d {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n > 1e-10 {
            Self::new(self.x / n, self.y / n, self.z / n)
        } else {
            *self
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn cwisemin(&self, other: &Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    pub fn cwisemax(&self, other: &Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    pub fn cwiseabs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl std::ops::Add for Vec3d {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Vec3d::add(&self, &other)
    }
}

impl std::ops::Sub for Vec3d {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Vec3d::sub(&self, &other)
    }
}

impl std::ops::Mul<f64> for Vec3d {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

/// 3x3 rotation matrix.
#[derive(Debug, Clone, Copy)]
pub struct Matrix3d {
    pub m: [[f64; 3]; 3],
}

impl Matrix3d {
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn rotation_x(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        }
    }

    pub fn rotation_y(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            m: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        }
    }

    pub fn rotation_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        Self { m: result }
    }

    pub fn mul_vec(&self, v: &Vec3d) -> Vec3d {
        Vec3d::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }
}

/// Transform from octree coordinates to world coordinates.
pub fn transform_to_world() -> Matrix3d {
    Matrix3d::rotation_z(OCTREE_ROT[2])
        .mul_matrix(&Matrix3d::rotation_y(OCTREE_ROT[1]))
        .mul_matrix(&Matrix3d::rotation_x(OCTREE_ROT[0]))
}

/// Transform from world coordinates to octree coordinates.
pub fn transform_to_octree() -> Matrix3d {
    Matrix3d::rotation_x(-OCTREE_ROT[0])
        .mul_matrix(&Matrix3d::rotation_y(-OCTREE_ROT[1]))
        .mul_matrix(&Matrix3d::rotation_z(-OCTREE_ROT[2]))
}

/// 3D axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABBf3 {
    pub min: Vec3d,
    pub max: Vec3d,
}

impl AABBf3 {
    pub fn new(min: Vec3d, max: Vec3d) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Vec3d]) -> Self {
        if points.is_empty() {
            return Self::new(Vec3d::zero(), Vec3d::zero());
        }
        let mut min = points[0];
        let mut max = points[0];
        for p in points.iter().skip(1) {
            min = min.cwisemin(p);
            max = max.cwisemax(p);
        }
        Self::new(min, max)
    }

    pub fn center(&self) -> Vec3d {
        Vec3d::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn size(&self) -> Vec3d {
        self.max.sub(&self.min)
    }

    pub fn max_extent(&self) -> f64 {
        let s = self.size();
        s.x.max(s.y).max(s.z)
    }
}

/// Test if a triangle intersects an AABB using SAT (Separating Axis Theorem).
///
/// Based on "Real-Time Collision Detection" by Christer Ericson, pp. 169-172.
pub fn triangle_aabb_intersects(a: &Vec3d, b: &Vec3d, c: &Vec3d, aabb: &AABBf3) -> bool {
    // Triangle AABB test
    let t_min = a.cwisemin(&b.cwisemin(c));
    let t_max = a.cwisemax(&b.cwisemax(c));

    if t_min.x >= aabb.max.x
        || t_max.x <= aabb.min.x
        || t_min.y >= aabb.max.y
        || t_max.y <= aabb.min.y
        || t_min.z >= aabb.max.z
        || t_max.z <= aabb.min.z
    {
        return false;
    }

    let center = aabb.center();
    let h = Vec3d::new(
        aabb.max.x - center.x,
        aabb.max.y - center.y,
        aabb.max.z - center.z,
    );

    // Edge vectors
    let t0 = b.sub(a);
    let t1 = c.sub(a);
    let t2 = c.sub(b);

    let ac = a.sub(&center);

    // Normal test
    let n = t0.cross(&t1);
    let s = n.dot(&ac);
    let r = h.dot(&n.cwiseabs());
    if s.abs() >= r {
        return false;
    }

    let at0 = t0.cwiseabs();
    let at1 = t1.cwiseabs();
    let at2 = t2.cwiseabs();

    let bc = b.sub(&center);
    let cc = c.sub(&center);

    // SAT tests for all 9 cross-product axes
    // eX x t[0]
    let d1 = t0.y * ac.z - t0.z * ac.y;
    let d2 = t0.y * cc.z - t0.z * cc.y;
    let tc = (d1 + d2) * 0.5;
    let r = (h.y * at0.z + h.z * at0.y).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eX x t[1]
    let d1 = t1.y * ac.z - t1.z * ac.y;
    let d2 = t1.y * bc.z - t1.z * bc.y;
    let tc = (d1 + d2) * 0.5;
    let r = (h.y * at1.z + h.z * at1.y).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eX x t[2]
    let d1 = t2.y * ac.z - t2.z * ac.y;
    let d2 = t2.y * bc.z - t2.z * bc.y;
    let tc = (d1 + d2) * 0.5;
    let r = (h.y * at2.z + h.z * at2.y).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eY x t[0]
    let d1 = t0.z * ac.x - t0.x * ac.z;
    let d2 = t0.z * cc.x - t0.x * cc.z;
    let tc = (d1 + d2) * 0.5;
    let r = (h.x * at0.z + h.z * at0.x).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eY x t[1]
    let d1 = t1.z * ac.x - t1.x * ac.z;
    let d2 = t1.z * bc.x - t1.x * bc.z;
    let tc = (d1 + d2) * 0.5;
    let r = (h.x * at1.z + h.z * at1.x).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eY x t[2]
    let d1 = t2.z * ac.x - t2.x * ac.z;
    let d2 = t2.z * bc.x - t2.x * bc.z;
    let tc = (d1 + d2) * 0.5;
    let r = (h.x * at2.z + h.z * at2.x).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eZ x t[0]
    let d1 = t0.x * ac.y - t0.y * ac.x;
    let d2 = t0.x * cc.y - t0.y * cc.x;
    let tc = (d1 + d2) * 0.5;
    let r = (h.y * at0.x + h.x * at0.y).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eZ x t[1]
    let d1 = t1.x * ac.y - t1.y * ac.x;
    let d2 = t1.x * bc.y - t1.y * bc.x;
    let tc = (d1 + d2) * 0.5;
    let r = (h.y * at1.x + h.x * at1.y).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // eZ x t[2]
    let d1 = t2.x * ac.y - t2.y * ac.x;
    let d2 = t2.x * bc.y - t2.y * bc.x;
    let tc = (d1 + d2) * 0.5;
    let r = (h.y * at2.x + h.x * at2.y).abs();
    if r + (tc - d1).abs() < tc.abs() {
        return false;
    }

    // No separating axis found - they intersect
    true
}

/// A single cube in the octree.
#[derive(Debug)]
pub struct Cube {
    /// Center position in world coordinates.
    pub center: Vec3d,

    /// Center position in octree coordinates (for debugging).
    #[cfg(debug_assertions)]
    pub center_octree: Vec3d,

    /// Children cubes (8 possible children, None if not subdivided).
    pub children: [Option<Box<Cube>>; 8],
}

impl Cube {
    pub fn new(center: Vec3d) -> Self {
        Self {
            center,
            #[cfg(debug_assertions)]
            center_octree: center,
            children: Default::default(),
        }
    }
}

/// Octree data structure for adaptive infill.
#[derive(Debug)]
pub struct Octree {
    /// Root cube of the octree.
    pub root_cube: Option<Box<Cube>>,

    /// Origin point of the octree.
    pub origin: Vec3d,

    /// Properties for cubes at each level.
    pub cubes_properties: Vec<CubeProperties>,
}

impl Octree {
    /// Create a new octree.
    pub fn new(origin: Vec3d, cubes_properties: Vec<CubeProperties>) -> Self {
        Self {
            root_cube: Some(Box::new(Cube::new(origin))),
            origin,
            cubes_properties,
        }
    }
}

/// Insert a triangle into the octree cube, subdividing as needed.
/// This is a free function to avoid borrow checker issues with Octree.
fn insert_triangle_into_cube(
    a: &Vec3d,
    b: &Vec3d,
    c: &Vec3d,
    cube: &mut Cube,
    bbox: &AABBf3,
    depth: usize,
    cubes_properties: &[CubeProperties],
) {
    if depth == 0 {
        return;
    }

    let new_depth = depth - 1;
    let edge_length = cubes_properties[new_depth].edge_length;

    for i in 0..8 {
        let child_center_dir = CHILD_CENTERS[i];

        // Calculate child bounding box (slightly expanded for numerical stability)
        let epsilon = 1e-6;
        let mut child_bbox = AABBf3::new(Vec3d::zero(), Vec3d::zero());

        for k in 0..3 {
            let dir = [
                child_center_dir[0],
                child_center_dir[1],
                child_center_dir[2],
            ][k];
            let center_k = match k {
                0 => cube.center.x,
                1 => cube.center.y,
                _ => cube.center.z,
            };
            let bbox_min_k = match k {
                0 => bbox.min.x,
                1 => bbox.min.y,
                _ => bbox.min.z,
            };
            let bbox_max_k = match k {
                0 => bbox.max.x,
                1 => bbox.max.y,
                _ => bbox.max.z,
            };

            if dir == -1.0 {
                match k {
                    0 => {
                        child_bbox.min.x = bbox_min_k;
                        child_bbox.max.x = center_k + epsilon;
                    }
                    1 => {
                        child_bbox.min.y = bbox_min_k;
                        child_bbox.max.y = center_k + epsilon;
                    }
                    _ => {
                        child_bbox.min.z = bbox_min_k;
                        child_bbox.max.z = center_k + epsilon;
                    }
                }
            } else {
                match k {
                    0 => {
                        child_bbox.min.x = center_k - epsilon;
                        child_bbox.max.x = bbox_max_k;
                    }
                    1 => {
                        child_bbox.min.y = center_k - epsilon;
                        child_bbox.max.y = bbox_max_k;
                    }
                    _ => {
                        child_bbox.min.z = center_k - epsilon;
                        child_bbox.max.z = bbox_max_k;
                    }
                }
            }
        }

        // Calculate child center
        let child_center = Vec3d::new(
            cube.center.x + child_center_dir[0] * (edge_length / 2.0),
            cube.center.y + child_center_dir[1] * (edge_length / 2.0),
            cube.center.z + child_center_dir[2] * (edge_length / 2.0),
        );

        // Check if triangle intersects child bbox
        if triangle_aabb_intersects(a, b, c, &child_bbox) {
            // Create child if it doesn't exist
            if cube.children[i].is_none() {
                cube.children[i] = Some(Box::new(Cube::new(child_center)));
            }

            // Recursively insert into child
            if new_depth > 0 {
                if let Some(ref mut child) = cube.children[i] {
                    insert_triangle_into_cube(
                        a,
                        b,
                        c,
                        child.as_mut(),
                        &child_bbox,
                        new_depth,
                        cubes_properties,
                    );
                }
            }
        }
    }
}

/// Context for filling a single layer.
struct FillContext {
    /// Z position of the layer.
    z_position: CoordF,

    /// Direction index (0, 1, or 2).
    direction: usize,

    /// Traversal order for this direction.
    traversal_order: [usize; 8],

    /// Cosine of the rotation angle.
    cos_a: CoordF,

    /// Sine of the rotation angle.
    sin_a: CoordF,

    /// Cube properties reference.
    cubes_properties: Vec<CubeProperties>,

    /// Temporary lines being built (keyed by address).
    temp_lines: Vec<Option<(Point, Point)>>,

    /// Output lines (completed).
    output_lines: Vec<(Point, Point)>,
}

impl FillContext {
    fn new(octree: &Octree, z: CoordF, direction: usize) -> Self {
        // Rotation angle for this direction
        let angle = match direction {
            0 => -std::f64::consts::FRAC_PI_4,
            1 => std::f64::consts::FRAC_PI_4,
            _ => 0.75 * PI,
        };

        // Calculate max tree depth for temp_lines size
        let max_depth = octree.cubes_properties.len();
        let max_addresses = 1 << (max_depth + 1); // 2^(depth+1)

        Self {
            z_position: z,
            direction,
            traversal_order: CHILD_TRAVERSAL_ORDER[direction],
            cos_a: angle.cos(),
            sin_a: angle.sin(),
            cubes_properties: octree.cubes_properties.clone(),
            temp_lines: vec![None; max_addresses],
            output_lines: Vec::new(),
        }
    }

    /// Rotate a 2D point by the context's rotation.
    fn rotate(&self, x: CoordF, y: CoordF) -> (CoordF, CoordF) {
        (
            x * self.cos_a - y * self.sin_a,
            x * self.sin_a + y * self.cos_a,
        )
    }
}

/// Generate infill lines by recursively traversing the octree.
fn generate_infill_lines_recursive(
    ctx: &mut FillContext,
    cube: &Cube,
    address: usize,
    depth: usize,
) {
    if depth == 0 {
        return;
    }

    let props = &ctx.cubes_properties[depth - 1];
    let z_diff = ctx.z_position - cube.center.z;
    let z_diff_abs = z_diff.abs();

    // Skip if we're outside this cube's Z range
    if z_diff_abs > props.height / 2.0 {
        return;
    }

    // Generate a line if we're within the line generation zone
    if z_diff_abs < props.line_z_distance {
        let zdist = props.line_z_distance;

        // Calculate line endpoints relative to cube center
        let from_x = 0.5 * props.diagonal_length * (zdist - z_diff_abs) / zdist;
        let from_y = props.line_xy_distance - (zdist + z_diff) / 2.0_f64.sqrt();
        let to_x = -from_x;
        let to_y = from_y;

        // Rotate the line
        let (from_x, from_y) = ctx.rotate(from_x, from_y);
        let (to_x, to_y) = ctx.rotate(to_x, to_y);

        // Translate to world coordinates
        let from_x = from_x + cube.center.x;
        let from_y = from_y + cube.center.y;
        let to_x = to_x + cube.center.x;
        let to_y = to_y + cube.center.y;

        // Convert to scaled coordinates
        let from = Point::new(scale(from_x), scale(from_y));
        let to = Point::new(scale(to_x), scale(to_y));

        // Either extend existing line or start new one
        let scaled_epsilon: Coord = 1000; // Match BambuStudio's threshold

        if let Some(ref mut last_line) = ctx.temp_lines[address] {
            // Check if we can extend the existing line
            let dx = (to.x - last_line.1.x).abs();
            let dy = (to.y - last_line.1.y).abs();

            if dx.max(dy) > scaled_epsilon {
                // Lines don't connect - emit the old one
                ctx.output_lines.push(*last_line);
                last_line.0 = from;
            }
            last_line.1 = to;
        } else {
            ctx.temp_lines[address] = Some((from, to));
        }
    }

    // Recurse into children
    let new_depth = depth - 1;
    let left_address = address * 2 + 1;
    let right_address = left_address + 1;

    // Copy traversal order to avoid borrowing ctx during iteration
    let traversal_order = ctx.traversal_order;

    let mut child_count = 0;
    for &child_idx in &traversal_order {
        if let Some(ref child) = cube.children[child_idx] {
            let addr = if child_count < 4 {
                left_address
            } else {
                right_address
            };
            generate_infill_lines_recursive(ctx, child.as_ref(), addr, new_depth);
        }
        child_count += 1;
    }
}

/// Build an octree from mesh triangles.
pub fn build_octree(
    triangles: &[(Vec3d, Vec3d, Vec3d)],
    overhang_triangles: &[(Vec3d, Vec3d, Vec3d)],
    line_spacing: CoordF,
    support_overhangs_only: bool,
) -> Option<Octree> {
    if triangles.is_empty() {
        return None;
    }

    // Calculate bounding box of all triangles
    let mut all_points: Vec<Vec3d> = Vec::with_capacity(triangles.len() * 3);
    for (a, b, c) in triangles {
        all_points.push(*a);
        all_points.push(*b);
        all_points.push(*c);
    }
    let bbox = AABBf3::from_points(&all_points);
    let max_extent = bbox.max_extent();

    if max_extent <= 0.0 {
        return None;
    }

    // Transform to octree coordinates
    let to_octree = transform_to_octree();
    let cube_center = bbox.center();

    // Create cube properties
    let cubes_properties = make_cubes_properties(max_extent, line_spacing);

    if cubes_properties.len() <= 1 {
        return None;
    }

    let mut octree = Octree::new(cube_center, cubes_properties);
    let max_depth = octree.cubes_properties.len() - 1;

    // Calculate root bounding box
    let edge_length_half = 0.5 * octree.cubes_properties.last().unwrap().edge_length;
    let diag_half = Vec3d::new(edge_length_half, edge_length_half, edge_length_half);
    let root_bbox = AABBf3::new(cube_center.sub(&diag_half), cube_center.add(&diag_half));

    // Up vector for overhang detection
    let up_vector = if support_overhangs_only {
        Some(to_octree.mul_vec(&Vec3d::new(0.0, 0.0, 1.0)))
    } else {
        None
    };

    // Insert triangles - we need to extract root to avoid borrow issues
    for (a, b, c) in triangles {
        // Skip if only supporting overhangs and this isn't an overhang
        if let Some(ref up) = up_vector {
            let n = (b.sub(a)).cross(&(c.sub(b)));
            if n.dot(up) <= 0.707 * n.norm() {
                continue;
            }
        }

        // Take root out, modify, put back to avoid borrow issues
        if let Some(mut root) = octree.root_cube.take() {
            insert_triangle_into_cube(
                a,
                b,
                c,
                root.as_mut(),
                &root_bbox,
                max_depth,
                &octree.cubes_properties,
            );
            octree.root_cube = Some(root);
        }
    }

    // Insert overhang triangles
    for (a, b, c) in overhang_triangles {
        if let Some(mut root) = octree.root_cube.take() {
            insert_triangle_into_cube(
                a,
                b,
                c,
                root.as_mut(),
                &root_bbox,
                max_depth,
                &octree.cubes_properties,
            );
            octree.root_cube = Some(root);
        }
    }

    // Transform octree to world coordinates
    let to_world = transform_to_world();
    if let Some(ref mut root) = octree.root_cube {
        transform_cube_center(root.as_mut(), &to_world);
    }
    octree.origin = to_world.mul_vec(&octree.origin);

    Some(octree)
}

/// Recursively transform cube centers to world coordinates.
fn transform_cube_center(cube: &mut Cube, rot: &Matrix3d) {
    #[cfg(debug_assertions)]
    {
        cube.center_octree = cube.center;
    }
    cube.center = rot.mul_vec(&cube.center);

    for child in cube.children.iter_mut().flatten() {
        transform_cube_center(child.as_mut(), rot);
    }
}

/// Adaptive infill generator.
pub struct AdaptiveInfillGenerator {
    /// Configuration.
    config: AdaptiveInfillConfig,

    /// Octree built from mesh.
    octree: Option<Octree>,
}

impl AdaptiveInfillGenerator {
    /// Create a new generator with configuration.
    pub fn new(config: AdaptiveInfillConfig) -> Self {
        Self {
            config,
            octree: None,
        }
    }

    /// Create a generator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AdaptiveInfillConfig::default())
    }

    /// Build the octree from mesh triangles.
    ///
    /// Triangles should be in world coordinates (mm).
    pub fn build_from_triangles(&mut self, triangles: &[(Vec3d, Vec3d, Vec3d)]) {
        self.octree = build_octree(
            triangles,
            &[],
            self.config.line_spacing,
            self.config.support_overhangs_only,
        );
    }

    /// Build from overhang triangles as well.
    pub fn build_from_triangles_with_overhangs(
        &mut self,
        triangles: &[(Vec3d, Vec3d, Vec3d)],
        overhang_triangles: &[(Vec3d, Vec3d, Vec3d)],
    ) {
        self.octree = build_octree(
            triangles,
            overhang_triangles,
            self.config.line_spacing,
            self.config.support_overhangs_only,
        );
    }

    /// Generate infill lines for a layer at the given Z height.
    ///
    /// Returns raw infill lines before clipping to boundary.
    pub fn generate_lines(&self, z: CoordF) -> Vec<Polyline> {
        let octree = match &self.octree {
            Some(o) => o,
            None => return Vec::new(),
        };

        let root = match &octree.root_cube {
            Some(r) => r,
            None => return Vec::new(),
        };

        let max_depth = octree.cubes_properties.len();

        // Generate lines for all 3 directions
        let mut all_lines: Vec<(Point, Point)> = Vec::new();

        for direction in 0..3 {
            let mut ctx = FillContext::new(octree, z, direction);

            // Generate lines recursively
            generate_infill_lines_recursive(&mut ctx, root.as_ref(), 0, max_depth);

            // Collect output lines
            all_lines.extend(&ctx.output_lines);

            // Collect remaining temp lines
            for temp_line in ctx.temp_lines.into_iter().flatten() {
                all_lines.push(temp_line);
            }
        }

        // Convert to polylines
        all_lines
            .into_iter()
            .map(|(a, b)| Polyline::from_points(vec![a, b]))
            .collect()
    }

    /// Generate infill for a region, clipped to the boundary.
    pub fn generate(&self, z: CoordF, boundary: &ExPolygon) -> Vec<Polyline> {
        let lines = self.generate_lines(z);

        if lines.is_empty() {
            return Vec::new();
        }

        // Clip lines to the boundary using intersection
        let mut result = Vec::new();

        for line in lines {
            // Convert polyline to polygon for intersection (temporary)
            // We need to clip each line segment to the boundary
            let clipped = clip_polyline_to_expolygon(&line, boundary);
            result.extend(clipped);
        }

        result
    }

    /// Get a reference to the octree (for debugging/visualization).
    pub fn octree(&self) -> Option<&Octree> {
        self.octree.as_ref()
    }

    /// Check if octree is built.
    pub fn is_ready(&self) -> bool {
        self.octree.is_some()
    }
}

/// Clip a polyline to an ExPolygon boundary.
fn clip_polyline_to_expolygon(polyline: &Polyline, boundary: &ExPolygon) -> Vec<Polyline> {
    // For now, use a simple implementation that checks each segment
    // A more sophisticated implementation would use Clipper's intersection_pl

    let points = polyline.points();
    if points.len() < 2 {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current_segment: Vec<Point> = Vec::new();

    for i in 0..points.len() {
        let p = &points[i];
        let inside = point_in_expolygon(p, boundary);

        if inside {
            current_segment.push(p.clone());
        } else {
            if current_segment.len() >= 2 {
                result.push(Polyline::from_points(current_segment));
            }
            current_segment = Vec::new();
        }
    }

    // Don't forget the last segment
    if current_segment.len() >= 2 {
        result.push(Polyline::from_points(current_segment));
    }

    result
}

/// Check if a point is inside an ExPolygon.
fn point_in_expolygon(point: &Point, expoly: &ExPolygon) -> bool {
    // Point must be inside outer contour
    if !point_in_polygon(point, &expoly.contour) {
        return false;
    }

    // Point must not be inside any hole
    for hole in &expoly.holes {
        if point_in_polygon(point, hole) {
            return false;
        }
    }

    true
}

/// Check if a point is inside a polygon (using ray casting).
fn point_in_polygon(point: &Point, polygon: &crate::geometry::Polygon) -> bool {
    let points = polygon.points();
    let n = points.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let x = point.x;
    let y = point.y;

    let mut j = n - 1;
    for i in 0..n {
        let xi = points[i].x;
        let yi = points[i].y;
        let xj = points[j].x;
        let yj = points[j].y;

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Result of adaptive infill generation.
#[derive(Debug, Clone)]
pub struct AdaptiveInfillResult {
    /// Generated polylines.
    pub polylines: Vec<Polyline>,

    /// Total length of infill in mm.
    pub total_length_mm: CoordF,
}

impl AdaptiveInfillResult {
    /// Create a new result.
    pub fn new(polylines: Vec<Polyline>) -> Self {
        // length() returns scaled units, convert to mm
        let total_length_mm = polylines.iter().map(|p| p.length()).sum();

        Self {
            polylines,
            total_length_mm,
        }
    }

    /// Check if any infill was generated.
    pub fn is_empty(&self) -> bool {
        self.polylines.is_empty()
    }

    /// Get the number of paths.
    pub fn path_count(&self) -> usize {
        self.polylines.len()
    }
}

/// Convenience function to generate adaptive infill.
pub fn generate_adaptive_infill(
    triangles: &[(Vec3d, Vec3d, Vec3d)],
    z: CoordF,
    boundary: &ExPolygon,
    config: AdaptiveInfillConfig,
) -> AdaptiveInfillResult {
    let mut generator = AdaptiveInfillGenerator::new(config);
    generator.build_from_triangles(triangles);
    let polylines = generator.generate(z, boundary);
    AdaptiveInfillResult::new(polylines)
}

/// Convenience function to generate adaptive infill with density.
pub fn generate_adaptive_infill_with_density(
    triangles: &[(Vec3d, Vec3d, Vec3d)],
    z: CoordF,
    boundary: &ExPolygon,
    density: CoordF,
    extrusion_width: CoordF,
) -> AdaptiveInfillResult {
    let config = AdaptiveInfillConfig::from_density(density, extrusion_width);
    generate_adaptive_infill(triangles, z, boundary, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_triangle() -> (Vec3d, Vec3d, Vec3d) {
        (
            Vec3d::new(0.0, 0.0, 0.0),
            Vec3d::new(10.0, 0.0, 0.0),
            Vec3d::new(5.0, 10.0, 5.0),
        )
    }

    fn make_test_cube_triangles() -> Vec<(Vec3d, Vec3d, Vec3d)> {
        // A simple cube from 0,0,0 to 10,10,10
        let s = 10.0;
        vec![
            // Bottom face
            (
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(s, 0.0, 0.0),
                Vec3d::new(s, s, 0.0),
            ),
            (
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(s, s, 0.0),
                Vec3d::new(0.0, s, 0.0),
            ),
            // Top face
            (
                Vec3d::new(0.0, 0.0, s),
                Vec3d::new(s, s, s),
                Vec3d::new(s, 0.0, s),
            ),
            (
                Vec3d::new(0.0, 0.0, s),
                Vec3d::new(0.0, s, s),
                Vec3d::new(s, s, s),
            ),
            // Front face
            (
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(s, 0.0, s),
                Vec3d::new(s, 0.0, 0.0),
            ),
            (
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(0.0, 0.0, s),
                Vec3d::new(s, 0.0, s),
            ),
            // Back face
            (
                Vec3d::new(0.0, s, 0.0),
                Vec3d::new(s, s, 0.0),
                Vec3d::new(s, s, s),
            ),
            (
                Vec3d::new(0.0, s, 0.0),
                Vec3d::new(s, s, s),
                Vec3d::new(0.0, s, s),
            ),
            // Left face
            (
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(0.0, s, 0.0),
                Vec3d::new(0.0, s, s),
            ),
            (
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(0.0, s, s),
                Vec3d::new(0.0, 0.0, s),
            ),
            // Right face
            (
                Vec3d::new(s, 0.0, 0.0),
                Vec3d::new(s, 0.0, s),
                Vec3d::new(s, s, s),
            ),
            (
                Vec3d::new(s, 0.0, 0.0),
                Vec3d::new(s, s, s),
                Vec3d::new(s, s, 0.0),
            ),
        ]
    }

    #[test]
    fn test_vec3d_operations() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert!((sum.x - 5.0).abs() < 1e-10);
        assert!((sum.y - 7.0).abs() < 1e-10);
        assert!((sum.z - 9.0).abs() < 1e-10);

        let dot = a.dot(&b);
        assert!((dot - 32.0).abs() < 1e-10);

        let cross = a.cross(&b);
        assert!((cross.x - (-3.0)).abs() < 1e-10);
        assert!((cross.y - 6.0).abs() < 1e-10);
        assert!((cross.z - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_matrix3d_rotation() {
        let m = Matrix3d::rotation_z(std::f64::consts::FRAC_PI_2);
        let v = Vec3d::new(1.0, 0.0, 0.0);
        let rotated = m.mul_vec(&v);

        assert!(rotated.x.abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
        assert!(rotated.z.abs() < 1e-10);
    }

    #[test]
    fn test_cube_properties() {
        let props = CubeProperties::new(2.0);

        assert!((props.edge_length - 2.0).abs() < 1e-10);
        assert!((props.height - 2.0 * 3.0_f64.sqrt()).abs() < 1e-10);
        assert!((props.diagonal_length - 2.0 * 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_make_cubes_properties() {
        let props = make_cubes_properties(10.0, 2.0);

        assert!(!props.is_empty());
        assert!((props[0].edge_length - 4.0).abs() < 1e-10); // 2.0 * 2.0

        // Edge lengths should double each level
        for i in 1..props.len() {
            let ratio = props[i].edge_length / props[i - 1].edge_length;
            assert!((ratio - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_triangle_aabb_intersects_inside() {
        let a = Vec3d::new(0.0, 0.0, 0.0);
        let b = Vec3d::new(1.0, 0.0, 0.0);
        let c = Vec3d::new(0.5, 1.0, 0.0);

        let aabb = AABBf3::new(Vec3d::new(-1.0, -1.0, -1.0), Vec3d::new(2.0, 2.0, 1.0));

        assert!(triangle_aabb_intersects(&a, &b, &c, &aabb));
    }

    #[test]
    fn test_triangle_aabb_intersects_outside() {
        let a = Vec3d::new(0.0, 0.0, 0.0);
        let b = Vec3d::new(1.0, 0.0, 0.0);
        let c = Vec3d::new(0.5, 1.0, 0.0);

        let aabb = AABBf3::new(Vec3d::new(10.0, 10.0, 10.0), Vec3d::new(20.0, 20.0, 20.0));

        assert!(!triangle_aabb_intersects(&a, &b, &c, &aabb));
    }

    #[test]
    fn test_build_octree() {
        let triangles = vec![make_test_triangle()];
        let octree = build_octree(&triangles, &[], 2.0, false);

        assert!(octree.is_some());
        let octree = octree.unwrap();
        assert!(octree.root_cube.is_some());
        assert!(!octree.cubes_properties.is_empty());
    }

    #[test]
    fn test_build_octree_cube() {
        let triangles = make_test_cube_triangles();
        let octree = build_octree(&triangles, &[], 2.0, false);

        assert!(octree.is_some());
        let octree = octree.unwrap();

        // Should have subdivided to some depth
        let root = octree.root_cube.as_ref().unwrap();
        let has_children = root.children.iter().any(|c| c.is_some());
        assert!(has_children);
    }

    #[test]
    fn test_adaptive_config_from_density() {
        let config = AdaptiveInfillConfig::from_density(0.2, 0.45);

        assert!((config.density_line_spacing() - 2.25).abs() < 1e-10); // 0.45 / 0.2
        assert!((config.extrusion_width - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_generator_creation() {
        let gen = AdaptiveInfillGenerator::with_defaults();
        assert!(!gen.is_ready());
    }

    #[test]
    fn test_generator_build() {
        let triangles = make_test_cube_triangles();
        let mut gen = AdaptiveInfillGenerator::with_defaults();
        gen.build_from_triangles(&triangles);

        assert!(gen.is_ready());
        assert!(gen.octree().is_some());
    }

    #[test]
    fn test_generator_generate_lines() {
        let triangles = make_test_cube_triangles();
        let mut gen = AdaptiveInfillGenerator::new(AdaptiveInfillConfig {
            line_spacing: 2.0,
            ..Default::default()
        });
        gen.build_from_triangles(&triangles);

        // Generate lines at middle of cube
        let lines = gen.generate_lines(5.0);

        // Should generate some lines
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_point_in_polygon() {
        use crate::geometry::Polygon;

        let square = Polygon::from_points(vec![
            Point::new(0, 0),
            Point::new(scale(10.0), 0),
            Point::new(scale(10.0), scale(10.0)),
            Point::new(0, scale(10.0)),
        ]);

        // Point inside
        let inside = Point::new(scale(5.0), scale(5.0));
        assert!(point_in_polygon(&inside, &square));

        // Point outside
        let outside = Point::new(scale(15.0), scale(15.0));
        assert!(!point_in_polygon(&outside, &square));
    }

    #[test]
    fn test_adaptive_infill_result() {
        let polylines = vec![
            Polyline::from_points(vec![Point::new(0, 0), Point::new(scale(10.0), 0)]),
            Polyline::from_points(vec![
                Point::new(0, scale(1.0)),
                Point::new(scale(10.0), scale(1.0)),
            ]),
        ];

        let result = AdaptiveInfillResult::new(polylines);

        assert_eq!(result.path_count(), 2);
        assert!(!result.is_empty());
        assert!(result.total_length_mm > 0.0);
    }

    #[test]
    fn test_transform_to_world_and_back() {
        let to_world = transform_to_world();
        let to_octree = transform_to_octree();

        let v = Vec3d::new(1.0, 2.0, 3.0);

        // Transform to octree then back to world should give original
        let v_octree = to_octree.mul_vec(&v);
        let v_back = to_world.mul_vec(&v_octree);

        assert!((v.x - v_back.x).abs() < 1e-10);
        assert!((v.y - v_back.y).abs() < 1e-10);
        assert!((v.z - v_back.z).abs() < 1e-10);
    }
}

impl AdaptiveInfillConfig {
    /// Get the effective line spacing based on density.
    pub fn density_line_spacing(&self) -> CoordF {
        self.line_spacing
    }
}
