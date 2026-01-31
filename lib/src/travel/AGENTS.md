# Travel Module

## Overview

The Travel module provides travel path planning and optimization, primarily through the AvoidCrossingPerimeters algorithm. This algorithm routes travel moves (non-extruding movements) around perimeter walls to prevent the nozzle from crossing over already-printed surfaces, which can leave visible marks.

## libslic3r Mapping

| Rust File | C++ File | Description |
|-----------|----------|-------------|
| `mod.rs` | `GCode/AvoidCrossingPerimeters.hpp`, `GCode/AvoidCrossingPerimeters.cpp` | Travel path planning to avoid crossing perimeters |

## Key Concepts

### Why Avoid Crossing Perimeters?

When the print head moves without extruding (travel moves), crossing over already-printed perimeter walls can cause several issues:

1. **Stringing**: Residual filament can be dragged across the surface
2. **Surface marks**: The nozzle can leave scratches or indentations
3. **Layer defects**: On lower layers, crossing can displace material

### Algorithm Overview

```
Direct travel: A ────────────────────────────► B
               (crosses perimeter wall)

Avoiding:      A ──┐                      ┌──► B
                   │     Perimeter        │
                   │    ┌────────┐       │
                   └────┤        ├───────┘
                        │        │
                        └────────┘
```

The algorithm:

1. **Intersection Detection**: Find where the direct travel line crosses boundary polygons
2. **Entry/Exit Points**: Identify where to enter and exit each boundary
3. **Path Routing**: Walk around the boundary (shortest direction)
4. **Path Simplification**: Remove unnecessary intermediate points
5. **Detour Validation**: Ensure the detour isn't too long

### Boundary Types

Two types of boundaries are supported:

- **Internal**: Boundaries within a single object (perimeter walls)
- **External**: Boundaries between multiple objects (for multi-object prints)

## Primary Types

### `TravelConfig`

Configuration for travel path planning.

```rust
pub struct TravelConfig {
    pub enabled: bool,              // Enable avoid crossing
    pub max_detour_percent: f64,    // Max detour as % of direct distance
    pub max_detour_absolute: i64,   // Max absolute detour (scaled units)
    pub grid_resolution: i64,       // EdgeGrid resolution
    pub boundary_offset: i64,       // Offset for boundary points
}
```

### `AvoidCrossingPerimeters`

The main travel planner.

```rust
pub struct AvoidCrossingPerimeters {
    config: TravelConfig,
    use_external: bool,
    internal: Boundary,
    external: Boundary,
    perimeter_spacing: i64,
}
```

### `TravelResult`

Result of travel planning.

```rust
pub struct TravelResult {
    pub path: Polyline,            // The travel path
    pub original_crossings: usize, // Crossings in direct path
    pub path_modified: bool,       // Whether path was changed
    pub wipe_disabled: bool,       // Whether wipe should be disabled
}
```

## Key Operations

### Initialization

```rust
// Create travel planner
let config = TravelConfig::default();
let mut avoid = AvoidCrossingPerimeters::new(config);

// Initialize for a layer
let boundaries = layer.perimeter_boundaries();
let perimeter_spacing = 400_000; // 0.4mm
avoid.init_layer(&boundaries, perimeter_spacing);

// For multi-object: also init external boundaries
avoid.init_external_boundaries(&external_boundaries);
```

### Planning Travel

```rust
let start = current_position;
let end = next_extrusion_start;

let result = avoid.travel_to(&start, &end);

if result.path_modified {
    // Use result.path instead of direct travel
    for point in result.path.points() {
        // Generate G0 moves
    }
}
```

### Per-Move Modifiers

```rust
// Use external boundaries for all moves
avoid.use_external_mp(true);

// Use external boundaries for next move only
avoid.use_external_mp_once();

// Disable for next move (e.g., first move of layer)
avoid.disable_once();

// Reset modifiers (called automatically after travel_to)
avoid.reset_once_modifiers();
```

## Algorithm Details

### Intersection Finding

Uses the `EdgeGrid` spatial structure to efficiently find where the travel line crosses boundary polygons:

```rust
let intersections = boundary.grid.find_intersections(&start, &end);
// Returns: Vec<Intersection> sorted by distance along line
```

### Path Routing

For each boundary crossing:

1. Find entry point (first intersection with boundary)
2. Find exit point (last intersection with same boundary)
3. Determine shortest direction (forward or backward around polygon)
4. Walk around boundary, collecting offset vertex positions
5. Add entry/exit points to path

### Direction Selection

```
Polygon perimeter: A ─► B ─► C ─► D ─► A
                   └──────────────────┘

If entering at segment 0 (A-B) and exiting at segment 2 (C-D):
  Forward:  A → B → C (length = 2 segments)
  Backward: A → D → C (length = 2 segments, but wrap-around)

Choose the direction with shorter total distance.
```

### Point Offsetting

Points are offset slightly inward from the boundary to ensure the travel path is fully inside/outside:

```rust
// Offset entry/exit points along edge normal
let offset_point = self.offset_point_inward(
    boundary,
    contour_idx,
    segment_idx, 
    &intersection_point,
);

// Offset vertices using averaged normals of adjacent edges
let offset_vertex = self.offset_vertex(boundary, poly_idx, vertex_idx);
```

### Path Simplification

After building the full path around boundaries, unnecessary points are removed:

```rust
// For each point, try to skip forward to furthest point
// that doesn't create new boundary crossings
for try_idx in (current + 2)..path.len() {
    if !boundary.grid.line_intersects_any(&current.point, &try_point) {
        // Can skip directly to try_idx
        best_next = try_idx;
    }
}
```

### Detour Validation

If the resulting path is too long compared to direct travel, fall back to direct:

```rust
let direct_length = distance(start, end);
let detour = path_length - direct_length;
let max_detour = direct_length * max_detour_percent / 100.0;

if detour > max_detour {
    // Use direct path instead
    return TravelResult::direct(start, end);
}
```

## Usage in G-code Generation

The travel planner integrates into the G-code generation pipeline:

```rust
// In GCodeGenerator
fn travel_to(&mut self, point: Point) {
    let result = self.avoid_crossing.travel_to(&self.position, &point);
    
    for (i, pt) in result.path.points().iter().enumerate() {
        if i == 0 { continue; } // Skip start point
        self.writer.travel_to(*pt);
    }
    
    self.position = point;
}
```

## Performance Considerations

- **EdgeGrid**: Uses spatial hashing for O(1) cell lookup instead of O(n) edge checks
- **Pre-computed distances**: Polygon perimeter distances cached for fast direction selection
- **Early exit**: Returns direct path immediately when no crossings detected
- **Simplification**: Removes redundant points to minimize G-code size

## Configuration Recommendations

| Setting | Default | Notes |
|---------|---------|-------|
| `enabled` | `true` | Disable for draft prints or speed priority |
| `max_detour_percent` | `200%` | 2x direct distance max; increase for better surface quality |
| `grid_resolution` | `100,000` (0.1mm) | Smaller = faster queries, more memory |
| `boundary_offset` | `1,000` (1μm) | Small offset to stay inside/outside boundary |

## Testing

```bash
cargo test travel
```

Key test cases:
- Disabled mode returns direct path
- No intersection returns direct path
- Intersection triggers path modification
- disable_once works for single move
- Empty boundaries handled correctly
- Path simplification reduces point count

## Future Improvements

1. **Better detour calculation**: Consider actual printed material, not just geometry
2. **Wipe integration**: Coordinate with retraction/wipe sequences
3. **Multi-region handling**: Better support for travel across multiple regions
4. **Adaptive resolution**: Adjust grid resolution based on feature size