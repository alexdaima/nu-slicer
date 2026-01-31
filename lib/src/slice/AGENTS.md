# Slice Module

## Purpose

The `slice` module converts a 3D triangle mesh into a series of 2D layer slices. This is the fundamental operation of FDM slicing - transforming a solid 3D model into horizontal cross-sections that can be printed layer by layer.

## What This Module Does

1. **Mesh Slicing** (`mesh_slicer.rs`): Intersect mesh triangles with horizontal planes at each layer height
2. **Layer Management** (`layer.rs`): Store and organize slice data per layer
3. **Slicer Orchestration** (`slicer.rs`): High-level slicing API and configuration
4. **Slicing Parameters** (`slicing_params.rs`): Layer height, raft, variable layer height settings
5. **Surface Classification** (`surface.rs`): Identify top, bottom, internal, and bridge surfaces

## libslic3r Reference

| Rust File | C++ File(s) | Description |
|-----------|-------------|-------------|
| `slicer.rs` | `Slicing.hpp/cpp` | High-level slicing orchestration |
| `mesh_slicer.rs` | `TriangleMeshSlicer.hpp/cpp` | Mesh-plane intersection |
| `layer.rs` | `Layer.hpp/cpp` | Layer data structure |
| `slicing_params.rs` | `Slicing.hpp` | Slicing configuration |
| `surface.rs` | `Surface.hpp/cpp`, `SurfaceCollection.hpp/cpp` | Surface types |

### Key C++ Classes

- **`TriangleMeshSlicer`**: Core slicing algorithm
- **`Layer`**: Container for layer geometry
- **`LayerRegion`**: Per-region data within a layer
- **`Surface`**: Classified polygon surface
- **`SurfaceCollection`**: Collection of surfaces

## Slicing Algorithm

### Overview

```
TriangleMesh
    │
    ▼
For each layer Z height:
    │
    ▼
┌─────────────────────────────────────┐
│ Find triangles intersecting Z plane │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Compute intersection line segments  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Chain segments into closed contours │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Identify holes vs outer contours    │
└─────────────────────────────────────┘
    │
    ▼
ExPolygon[] for this layer
```

### Triangle-Plane Intersection

For each triangle, determine how it intersects the slicing plane at height Z:

1. **Above plane**: All vertices Z > slice_z → no intersection
2. **Below plane**: All vertices Z < slice_z → no intersection  
3. **Intersecting**: Some vertices above, some below → compute intersection line

```rust
// Find where triangle edge crosses Z plane
fn intersect_edge(p1: Point3F, p2: Point3F, z: f64) -> Option<PointF> {
    if (p1.z - z) * (p2.z - z) >= 0.0 {
        return None;  // Both on same side
    }
    let t = (z - p1.z) / (p2.z - p1.z);
    Some(PointF {
        x: p1.x + t * (p2.x - p1.x),
        y: p1.y + t * (p2.y - p1.y),
    })
}
```

### Contour Chaining

The intersection produces unordered line segments. These must be chained into closed contours:

1. Build a map of segment endpoints
2. Start from any segment, follow connected segments
3. Continue until returning to start point
4. Repeat for remaining segments

### Layer Heights

```
┌─────────────────────────────────────┐  ← Top of object
│                                     │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  ← Layer N (Z = N × layer_height)
│                                     │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  ← Layer N-1
│                                     │
│          ...                        │
│                                     │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  ← Layer 1 (first_layer_height)
│                                     │
└─────────────────────────────────────┘  ← Build plate (Z = 0)
```

## Key Types

### SlicingParams

```rust
pub struct SlicingParams {
    pub layer_height: f64,           // Regular layer height (mm)
    pub first_layer_height: f64,     // First layer (often thicker)
    pub object_height: f64,          // Total object height
    pub raft_layers: u32,            // Number of raft layers
    pub adaptive_slicing: bool,      // Variable layer heights
}
```

### Layer

```rust
pub struct Layer {
    pub id: usize,                   // Layer index (0 = first)
    pub print_z: f64,                // Z height for printing
    pub slice_z: f64,                // Z height for slicing
    pub height: f64,                 // This layer's thickness
    pub slices: Vec<ExPolygon>,      // Slice geometry
    pub regions: Vec<LayerRegion>,   // Per-region data
}
```

### LayerRegion

```rust
pub struct LayerRegion {
    pub perimeters: ExtrusionEntityCollection,
    pub fills: ExtrusionEntityCollection,
    pub thin_fills: ExtrusionEntityCollection,
    pub surfaces: SurfaceCollection,
}
```

### Surface

```rust
pub struct Surface {
    pub expolygon: ExPolygon,
    pub surface_type: SurfaceType,
}

pub enum SurfaceType {
    Top,              // Top visible surface
    Bottom,           // Bottom surface on bed or support
    BottomBridge,     // Bottom surface that bridges
    Internal,         // Internal (not visible)
    InternalSolid,    // Internal but needs solid infill
    InternalBridge,   // Internal bridging surface
    InternalVoid,     // Internal void (no infill needed)
}
```

## Surface Classification

Surfaces are classified by analyzing neighboring layers:

| Type | Condition |
|------|-----------|
| Top | No layer above, or layer above doesn't cover this area |
| Bottom | No layer below, or layer below doesn't cover this area |
| Bridge | Bottom surface with gap below (needs bridging) |
| Internal | Covered above and below |
| InternalSolid | Internal but within top/bottom shell count |

## File Structure

```
slicer/src/slice/
├── AGENTS.md           # This file
├── mod.rs              # Module exports
├── slicer.rs           # High-level Slicer type
├── mesh_slicer.rs      # Mesh-plane intersection
├── layer.rs            # Layer and LayerRegion types
├── slicing_params.rs   # SlicingParams configuration
└── surface.rs          # Surface and SurfaceType
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_height` | 0.2mm | Regular layer thickness |
| `first_layer_height` | 0.3mm | First layer (bed adhesion) |
| `raft_layers` | 0 | Raft layers below object |
| `slice_closing_radius` | 0.049mm | Gap closing for slices |
| `resolution` | 0.0125mm | Slice polygon simplification |

## Slice Position

Slices are taken at the **middle** of each layer by default:

```
slice_z = print_z - layer_height / 2
```

This gives the most representative cross-section for each layer.

## Edge Cases

### Horizontal Faces

Triangles parallel to the slicing plane (horizontal faces) require special handling:
- If exactly at slice_z, they contribute no intersection
- Numerical precision issues can cause missing/duplicate segments

### Thin Features

Features thinner than layer_height may be missed entirely. The `slice_closing_radius` parameter helps by closing small gaps in slices.

### Non-Manifold Meshes

Non-manifold edges (shared by >2 triangles) can cause:
- Unclosed contours
- Self-intersecting polygons
- Missing regions

## Performance Considerations

1. **Spatial indexing**: Use AABB tree to quickly find triangles at each Z
2. **Parallel slicing**: Each layer can be sliced independently
3. **Contour caching**: Reuse contour chains where geometry is identical

## Dependencies

- `crate::mesh::TriangleMesh` - Input mesh
- `crate::geometry::{ExPolygon, Polygon, Point, PointF}` - Output geometry
- `crate::clipper` - Polygon operations for gap closing

## Related Modules

- `mesh/` - Provides input TriangleMesh
- `perimeter/` - Consumes layers, generates perimeters
- `infill/` - Fills layer regions
- `support/` - Uses layer data for support detection
- `bridge/` - Detects bridges between layers

## Adaptive Layer Heights

The `adaptive_heights.rs` module implements variable layer height computation based on mesh surface slope.

### Algorithm Overview

Based on Florens Wasserfall's paper "Adaptive Slicing for the FDM Process Revisited":

1. **Collect Faces**: Extract all mesh triangles with their Z-span and normal angles
2. **Sort by Z**: Order faces by minimum Z coordinate for efficient traversal
3. **Compute Heights**: For each layer, find intersecting faces and compute optimal height
4. **Apply Limits**: Clamp heights to min/max layer height bounds

### Key Types

```rust
pub struct AdaptiveHeightsConfig {
    pub min_layer_height: f64,   // Minimum allowed height (mm)
    pub max_layer_height: f64,   // Maximum allowed height (mm)
    pub layer_height: f64,       // Default/target height (mm)
    pub quality: f64,            // Quality factor (0.0-1.0)
    pub error_metric: SlopeErrorMetric,
}

pub struct FaceZ {
    pub z_span: (f64, f64),      // (min_z, max_z)
    pub n_cos: f64,              // |normal.z| (0=vertical, 1=horizontal)
    pub n_sin: f64,              // sqrt(nx² + ny²)
}

pub struct AdaptiveLayerHeight {
    pub bottom_z: f64,
    pub top_z: f64,
    pub slice_z: f64,
    pub height: f64,
}
```

### Error Metrics

Four different formulas for computing layer height from surface slope:

| Metric | Formula | Best For |
|--------|---------|----------|
| TriangleArea | `min(clamped, 1.44 × dev × √(sin/cos))` | Default, balanced |
| TopographicDistance | `dev × sin/cos` | Cura-style |
| SurfaceRoughness | `dev × sin` | Constant surface stepping |
| Wasserfall | `dev / (0.184 + 0.5 × cos)` | Original paper |

### Quality Factor

Quality (0.0-1.0) controls the tradeoff between detail and speed:

- **0.0**: Highest quality, thinnest layers, most detail on slopes
- **0.5**: Balanced (uses default layer height as target)
- **1.0**: Fastest printing, thickest layers allowed

```rust
let max_deviation = if quality < 0.5 {
    lerp(min_height, layer_height, 2.0 * quality)
} else {
    lerp(max_height, layer_height, 2.0 * (1.0 - quality))
};
```

### Usage

```rust
let config = AdaptiveHeightsConfig::from_slicing_params(&params)
    .with_quality(0.3);
let mut slicer = AdaptiveSlicing::new(config);
let heights = slicer.compute_layer_heights(&mesh, first_layer_height);
```

### Convenience Functions

```rust
// Basic adaptive heights
let heights = compute_adaptive_heights(&mesh, &params);

// With quality setting
let heights = compute_adaptive_heights_with_quality(&mesh, &params, 0.3);
```

Reference: BambuStudio `SlicingAdaptive.cpp`

## Testing Strategy

1. **Unit tests**: Intersection math, contour chaining
2. **Simple shapes**: Cube, cylinder, sphere - verify layer counts
3. **Complex shapes**: 3DBenchy - compare with BambuStudio
4. **Edge cases**: Very thin features, horizontal faces
5. **Performance**: Benchmark with large meshes