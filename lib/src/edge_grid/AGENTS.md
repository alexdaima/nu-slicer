# EdgeGrid Module

## Overview

The EdgeGrid module provides a spatial acceleration structure for efficient queries on polygon edges. It divides the bounding box of a set of polygons into a regular grid of cells and stores which polygon segments pass through each cell, enabling fast intersection testing and closest-point queries.

## libslic3r Mapping

| Rust File | C++ File | Description |
|-----------|----------|-------------|
| `mod.rs` | `EdgeGrid.hpp`, `EdgeGrid.cpp` | Spatial grid for polygon edges |

## Key Concepts

### Grid Structure

The EdgeGrid divides space into a regular grid based on a configurable resolution:

```
┌────┬────┬────┬────┐
│    │    │ E1 │    │
├────┼────┼────┼────┤
│    │ E1 │ E2 │    │   E1, E2 = polygon edges
├────┼────┼────┼────┤       stored in cells they
│ E1 │    │    │ E2 │       pass through
├────┼────┼────┼────┤
│    │    │    │    │
└────┴────┴────┴────┘
```

Each cell stores indices to the polygon edges (contour_idx, segment_idx) that pass through it.

### Contour

A `Contour` represents a sequence of connected points:
- **Closed contour**: A polygon where the last point connects to the first
- **Open contour**: A polyline with distinct start and end points

### Cell Traversal

When querying a line against the grid, we use a Bresenham-like algorithm to visit all cells the line passes through, then check only the edges in those cells.

## Primary Types

### `Contour`

Represents a polygon or polyline stored in the grid.

```rust
pub struct Contour {
    points: Vec<Point>,
    open: bool,
}
```

### `EdgeGrid`

The main spatial acceleration structure.

```rust
pub struct EdgeGrid {
    bbox: BoundingBox,
    resolution: i64,
    rows: usize,
    cols: usize,
    contours: Vec<Contour>,
    cell_data: Vec<(usize, usize)>,  // (contour_idx, segment_idx)
    cells: Vec<Cell>,
    signed_distance_field: Vec<f32>,
}
```

### `ClosestPointResult`

Result of a closest-point query.

```rust
pub struct ClosestPointResult {
    pub contour_idx: usize,
    pub start_point_idx: usize,
    pub distance: f64,
    pub t: f64,
    pub point: Point,
}
```

### `Intersection`

Result of a line-polygon intersection query.

```rust
pub struct Intersection {
    pub contour_idx: usize,
    pub segment_idx: usize,
    pub point: Point,
    pub distance: f64,
}
```

## Key Operations

### Grid Creation

```rust
// From polygons
let grid = EdgeGrid::from_polygons(&polygons, resolution);

// From a single polygon
let grid = EdgeGrid::from_polygon(&polygon, resolution);

// From polylines
let mut grid = EdgeGrid::new();
grid.create_from_polylines(&polylines, resolution);

// From mixed
grid.create_from_mixed(&polygons, &polylines, resolution);
```

### Line Intersection

```rust
// Check if line intersects any edge
let intersects = grid.line_intersects_any(&p1, &p2);

// Find all intersection points
let intersections = grid.find_intersections(&p1, &p2);
// Returns sorted by distance along the line
```

### Closest Point Query

```rust
let result = grid.closest_point(&query_point, search_radius);
if result.is_valid() {
    println!("Closest point: {:?}", result.point);
    println!("Distance: {}", result.distance);
}
```

### Point-in-Polygon Test

```rust
let inside = grid.point_inside(&point);
```

### Signed Distance Field

```rust
// Calculate SDF for whole grid
grid.calculate_sdf();

// Query SDF with bilinear interpolation
let dist = grid.signed_distance_bilinear(&point);
// Negative = inside, Positive = outside
```

## Algorithm Details

### Cell Traversal Algorithm

The grid uses a modified Bresenham algorithm to enumerate cells a line segment passes through:

1. Convert endpoints to cell coordinates
2. Walk from start cell to end cell
3. At each step, move horizontally, vertically, or diagonally based on error accumulator

This ensures all cells that could contain an intersection are visited.

### Closest Point on Segment

For a query point Q and segment P1-P2:

1. Project Q onto the infinite line through P1-P2
2. Clamp the parameter t to [0, 1]
3. Return the point at parameter t

```
t = dot(Q - P1, P2 - P1) / |P2 - P1|²
t_clamped = clamp(t, 0, 1)
closest = P1 + t_clamped * (P2 - P1)
```

### Resolution Selection

The resolution parameter controls the cell size:
- **Smaller resolution**: More cells, faster queries, more memory
- **Larger resolution**: Fewer cells, slower queries, less memory

A good default is 1-2x the typical edge length or the perimeter spacing.

## Usage in Slicer

The EdgeGrid is primarily used by:

1. **AvoidCrossingPerimeters**: To find travel paths that don't cross perimeter walls
2. **Support generation**: For collision detection with the model
3. **Mesh operations**: For fast edge proximity queries

## Performance Considerations

- Grid creation is O(n) where n is total segment count
- Intersection queries are O(k) where k is segments in visited cells
- Memory usage is O(rows * cols + n)

For best performance:
- Choose resolution based on expected query distances
- Reuse grids for multiple queries on the same layer
- Use `line_intersects_any` when you only need a boolean result

## Testing

```bash
cargo test edge_grid
```

Key test cases:
- Grid creation from polygons
- Line intersection detection
- Intersection point finding
- Closest point queries
- Point-in-polygon tests
- Multiple polygon handling
- Open polyline support