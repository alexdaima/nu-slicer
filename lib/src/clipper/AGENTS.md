# Clipper Module

## Purpose

The `clipper` module provides polygon boolean operations and offsetting capabilities by wrapping the Clipper2 library. These operations are fundamental to nearly every stage of the slicing pipeline - from computing perimeter offsets to boolean operations between layers.

## What This Module Does

1. **Boolean Operations**: Union, intersection, difference, XOR of polygon sets
2. **Polygon Offsetting**: Grow (outset) and shrink (inset) polygons
3. **ExPolygon Support**: Operations that handle polygons with holes correctly
4. **Path Simplification**: Reduce point count while preserving shape
5. **Coordinate Conversion**: Convert between internal types and Clipper types

## libslic3r Reference

| Rust Function | C++ File(s) | Description |
|---------------|-------------|-------------|
| `union()` | `ClipperUtils.cpp`, `Clipper2Utils.cpp` | Union of polygons |
| `union_ex()` | `ClipperUtils.cpp` | Union returning ExPolygons |
| `intersection()` | `ClipperUtils.cpp` | Intersection of polygons |
| `difference()` | `ClipperUtils.cpp` | Difference (A - B) |
| `xor()` | `ClipperUtils.cpp` | Symmetric difference |
| `offset_polygon()` | `ClipperUtils.cpp` | Offset single polygon |
| `offset_polygons()` | `ClipperUtils.cpp` | Offset multiple polygons |
| `offset_expolygon()` | `ClipperUtils.cpp` | Offset ExPolygon |
| `grow()` | `ClipperUtils.cpp` | Positive offset (outward) |
| `shrink()` | `ClipperUtils.cpp` | Negative offset (inward) |

### Key C++ Functions

- **`union_ex()`**: Primary union operation returning ExPolygons
- **`diff_ex()`**: Difference returning ExPolygons
- **`intersection_ex()`**: Intersection returning ExPolygons
- **`offset()`** / **`offset_ex()`**: Polygon offsetting
- **`expand()`** / **`shrink()`**: Convenience wrappers for offset

## Operations

### Union

Combine overlapping polygons into one:

```
  ┌────┐
  │ A  │
  │  ┌─┼───┐        ┌────────┐
  └──┼─┘ B │   →    │        │
     │     │        │  A ∪ B │
     └─────┘        └────────┘
```

### Intersection

Keep only overlapping regions:

```
  ┌────┐
  │ A  │
  │  ┌─┼───┐        ┌──┐
  └──┼─┘ B │   →    │  │  A ∩ B
     │     │        └──┘
     └─────┘
```

### Difference

Remove B from A:

```
  ┌────┐
  │ A  │
  │  ┌─┼───┐        ┌──┐
  └──┼─┘ B │   →    │  │  A - B
     │     │        └──┘
     └─────┘
```

### XOR (Symmetric Difference)

Keep non-overlapping regions:

```
  ┌────┐
  │ A  │
  │  ┌─┼───┐        ┌──┐ ┌───┐
  └──┼─┘ B │   →    │  │ │   │  A ⊕ B
     │     │        └──┘ └───┘
     └─────┘
```

### Offset (Grow/Shrink)

Expand or contract polygon boundaries:

```
Original        Grow (+)        Shrink (-)
┌────────┐    ┌──────────┐      ┌──────┐
│        │    │          │      │      │
│        │    │          │      │      │
└────────┘    └──────────┘      └──────┘
```

## Key Types

### OffsetType

```rust
pub enum OffsetType {
    Positive,  // Grow outward
    Negative,  // Shrink inward
}
```

### OffsetJoinType

```rust
pub enum OffsetJoinType {
    Square,   // Sharp corners (fast)
    Round,    // Rounded corners
    Miter,    // Extended sharp corners
}
```

## API

### Boolean Operations

```rust
/// Union of polygons
pub fn union(polygons: &[Polygon]) -> Vec<Polygon>;

/// Union returning ExPolygons (with hole detection)
pub fn union_ex(polygons: &[ExPolygon]) -> Vec<ExPolygon>;

/// Intersection of two polygon sets
pub fn intersection(a: &[Polygon], b: &[Polygon]) -> Vec<Polygon>;

/// Difference: A - B
pub fn difference(a: &[Polygon], b: &[Polygon]) -> Vec<Polygon>;

/// Symmetric difference: (A - B) ∪ (B - A)
pub fn xor(a: &[Polygon], b: &[Polygon]) -> Vec<Polygon>;
```

### Offset Operations

```rust
/// Offset a single polygon
pub fn offset_polygon(polygon: &Polygon, delta: f64, join_type: OffsetJoinType) -> Vec<Polygon>;

/// Offset multiple polygons
pub fn offset_polygons(polygons: &[Polygon], delta: f64, join_type: OffsetJoinType) -> Vec<Polygon>;

/// Offset ExPolygon (handles holes correctly)
pub fn offset_expolygon(expolygon: &ExPolygon, delta: f64, join_type: OffsetJoinType) -> Vec<ExPolygon>;

/// Offset multiple ExPolygons
pub fn offset_expolygons(expolygons: &[ExPolygon], delta: f64, join_type: OffsetJoinType) -> Vec<ExPolygon>;
```

### Convenience Functions

```rust
/// Grow (positive offset)
pub fn grow(polygons: &[Polygon], delta: f64) -> Vec<Polygon>;

/// Shrink (negative offset)
pub fn shrink(polygons: &[Polygon], delta: f64) -> Vec<Polygon>;
```

## File Structure

```
slicer/src/clipper/
├── AGENTS.md    # This file
└── mod.rs       # Clipper2 wrapper implementation
```

## Coordinate Handling

Clipper2 uses 64-bit integers internally, matching our scaled coordinates:

```rust
// Our Point uses Coord (i64), same as Clipper
// No scaling conversion needed, just type conversion

fn point_to_clipper(p: Point) -> clipper2::Point<i64> {
    clipper2::Point::new(p.x, p.y)
}

fn clipper_to_point(p: clipper2::Point<i64>) -> Point {
    Point::new(p.x, p.y)
}
```

## Join Types

When offsetting, corners are handled differently:

### Square Join
```
Original     Offset
   │            ╱
   │           ╱
───┘        ──╱
            Sharp 45° corners
```

### Round Join
```
Original     Offset
   │           ╲
   │            ╲
───┘        ────╯
            Rounded corner
```

### Miter Join
```
Original     Offset
   │            │
   │            │
───┘        ────┘
            Extended corner
```

## Common Usage Patterns

### Perimeter Generation

```rust
// Generate perimeter offsets
let outer_perimeter = shrink(&[slice.contour.clone()], perimeter_width / 2.0);
let inner_perimeters = shrink(&outer_perimeter, perimeter_width);
```

### Infill Area

```rust
// Compute area for infill (inside perimeters)
let perimeter_area = union(&all_perimeters);
let infill_area = difference(&slice_area, &perimeter_area);
```

### Overhang Detection

```rust
// Find unsupported regions
let supported = intersection(&current_layer, &previous_layer);
let overhangs = difference(&current_layer, &supported);
```

### Support Generation

```rust
// Expand overhangs for support
let support_base = grow(&overhangs, support_xy_distance);
```

## Performance Considerations

1. **Batch operations**: Combine polygons before boolean ops when possible
2. **Simplification**: Use `simplify()` to reduce point count after complex operations
3. **Join type**: Square join is fastest, Round is slowest
4. **Polygon complexity**: Operations are O(n log n) in point count

## Dependencies

- `clipper2` crate - Rust bindings to Clipper2 library
- `crate::geometry::{Polygon, ExPolygon, Point}` - Internal geometry types

## Related Modules

- `geometry/` - Provides Polygon, ExPolygon, Point types
- `perimeter/` - Uses offset for shell generation
- `infill/` - Uses difference to compute fill regions
- `support/` - Uses boolean ops for overhang detection
- `slice/` - Uses union to merge slice contours

## Testing Strategy

1. **Unit tests**: Each boolean operation with simple shapes
2. **Round-trip tests**: Grow then shrink returns similar shape
3. **Hole handling**: ExPolygon operations preserve holes correctly
4. **Edge cases**: Touching polygons, self-intersecting shapes
5. **Numerical tests**: Verify results against known values

## Clipper2 vs Original Clipper

This module uses Clipper2 (the modern rewrite), which offers:
- Better performance
- Improved numerical stability
- Native 64-bit integer support
- Better polygon simplification

libslic3r uses both original Clipper and Clipper2 depending on the operation.

## Error Handling

Clipper operations can fail or produce unexpected results with:
- Self-intersecting polygons
- Degenerate (zero-area) polygons
- Extreme coordinate values

The wrapper handles these gracefully by:
- Returning empty results for invalid input
- Simplifying results to remove degeneracies
- Using appropriate numerical precision

## Future Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| Basic boolean ops | ✅ Done | Union, intersection, difference, XOR |
| Polygon offset | ✅ Done | Grow and shrink |
| ExPolygon support | ✅ Done | Handle polygons with holes |
| Path simplification | 🔄 Partial | Reduce point count |
| Minkowski sum | 📋 Planned | Advanced offset operations |
| Polygon decomposition | 📋 Planned | Split complex polygons |