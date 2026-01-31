# Infill Module

## Purpose

The `infill` module generates interior fill patterns for 3D printed objects. Infill provides internal structure, strength, and support for top surfaces while minimizing material usage and print time.

## What This Module Does

1. **Pattern Generation**: Create various infill patterns (rectilinear, grid, honeycomb, gyroid, etc.)
2. **Density Control**: Adjust infill density from 0-100%
3. **Angle Rotation**: Rotate infill pattern between layers for strength
4. **Region Filling**: Fill ExPolygon regions while respecting holes
5. **Path Connection**: Optionally connect infill lines to reduce travel moves

## libslic3r Reference

| Rust | C++ File(s) | Description |
|------|-------------|-------------|
| `mod.rs` | `Fill/Fill.hpp/cpp` | Main infill orchestration |
| Pattern implementations | `Fill/FillBase.hpp/cpp` | Base class for all patterns |
| Rectilinear | `Fill/FillRectilinear.cpp` | Line-based patterns |
| Grid | `Fill/FillRectilinear.cpp` | Cross-hatch pattern |
| Honeycomb | `Fill/FillHoneycomb.cpp` | Hexagonal pattern |
| Gyroid | `Fill/FillGyroid.cpp` | Gyroid TPMS pattern |
| Concentric | `Fill/FillConcentric.cpp` | Inward-spiraling pattern |
| Lightning | `Fill/FillLightning.cpp` | Tree-like sparse infill |
| 3D Honeycomb | `Fill/Fill3DHoneycomb.cpp` | 3D interlocking pattern |
| `adaptive.rs` | `Fill/FillAdaptive.cpp` | Variable density infill |
| `cross_hatch.rs` | `Fill/FillCrossHatch.cpp` | Alternating direction pattern |
| `honeycomb_3d.rs` | `Fill/Fill3DHoneycomb.cpp` | Truncated octahedron pattern |

### Key C++ Classes

- **`Fill`** (`Fill.hpp`): Factory and orchestrator for infill patterns
- **`FillBase`**: Abstract base class for all fill patterns
- **`FillParams`**: Parameters passed to fill algorithms
- **`FillRectilinear`**: Line and grid patterns
- **`FillHoneycomb`**: Hexagonal honeycomb
- **`FillGyroid`**: Gyroid triply periodic minimal surface
- **`FillConcentric`**: Concentric inward fill
- **`FillLightning`**: Sparse tree-like support infill

## Infill Patterns

### Rectilinear (Line)

Simple parallel lines at specified angle. Fast to print, moderate strength.

```
────────────────────
────────────────────
────────────────────
────────────────────
```

**Use case**: Fast prints, low material usage

### Grid

Two perpendicular sets of lines (rectilinear at 0° and 90°). Good all-around pattern.

```
│ │ │ │ │ │ │ │ │ │
─┼─┼─┼─┼─┼─┼─┼─┼─┼─
│ │ │ │ │ │ │ │ │ │
─┼─┼─┼─┼─┼─┼─┼─┼─┼─
│ │ │ │ │ │ │ │ │ │
```

**Use case**: General purpose, good strength in all directions

### Honeycomb

Hexagonal pattern providing excellent strength-to-weight ratio.

```
 ╱╲   ╱╲   ╱╲   ╱╲
╱  ╲ ╱  ╲ ╱  ╲ ╱  ╲
╲  ╱ ╲  ╱ ╲  ╱ ╲  ╱
 ╲╱   ╲╱   ╲╱   ╲╱
 ╱╲   ╱╲   ╱╲   ╱╲
```

**Use case**: Strong parts, optimal material efficiency

### Gyroid

3D mathematically-defined surface (triply periodic minimal surface). Excellent strength in all directions, good for flexible parts.

```
∿∿∿∿∿∿∿∿∿∿
∿∿∿∿∿∿∿∿∿∿  (waves vary by layer)
∿∿∿∿∿∿∿∿∿∿
```

**Use case**: Maximum strength, flexible parts, isotropic properties

### Concentric

Follows the perimeter shape inward. Good for transparent or thin-walled parts.

```
┌────────────────┐
│ ┌────────────┐ │
│ │ ┌────────┐ │ │
│ │ │ ┌────┐ │ │ │
│ │ │ └────┘ │ │ │
│ │ └────────┘ │ │
│ └────────────┘ │
└────────────────┘
```

**Use case**: Thin walls, transparent parts, consistent appearance

### Floating Concentric

Enhanced concentric pattern that detects "floating" sections (not supported by the layer below).
Used primarily for top solid surfaces where print quality is critical.

```
┌────────────────┐
│ ┌────────────┐ │  ← Supported (on perimeters)
│ │ ~~~~~~~~ │ │    ← Floating (over sparse infill)
│ │ │ ┌────┐ │ │ │
│ │ │ └────┘ │ │ │
│ │ └────────┘ │ │
│ └────────────┘ │
└────────────────┘
```

**Key features**:
- Detects which loop segments are over unsupported areas
- Can split paths at floating/supported transitions
- Prefers starting loops at non-floating positions (better seam quality)
- Enables different extrusion parameters for floating sections

**Use case**: Top surfaces, quality-critical visible surfaces

**Usage**:
```rust
// Direct API with floating area detection
let result = generate_floating_concentric(&fill_area, &floating_areas, spacing);

// Or via InfillGenerator with floating areas
let generator = InfillGenerator::new(config);
let result = generator.generate_with_floating_areas(&fill_area, &floating_areas, layer);
```

### Lightning

Tree-like sparse structure that only supports top surfaces. Minimal material usage.

```
      │
    ┌─┴─┐
  ┌─┘   └─┐
┌─┘       └─┐
```

**Use case**: Minimal infill, parts that don't need internal strength

## InfillGenerator Integration

All infill patterns are accessible through the main `InfillGenerator::generate()` method:

```rust
let config = InfillConfig {
    pattern: InfillPattern::CrossHatch,  // or Honeycomb3D, etc.
    density: 0.2,
    z_height: 0.4,      // Required for 3D patterns
    layer_height: 0.2,  // Required for 3D patterns
    ..Default::default()
};

let generator = InfillGenerator::new(config);
let result = generator.generate(&fill_area, layer_index);
```

### Pattern Support Matrix

| Pattern | Via InfillGenerator | Notes |
|---------|:------------------:|-------|
| Rectilinear | ✅ | Standard lines |
| Grid | ✅ | Cross lines |
| Honeycomb | ✅ | 2D hexagons |
| Gyroid | ✅ | TPMS surface |
| Concentric | ✅ | Inward loops |
| FloatingConcentric | ✅ | Concentric with floating detection* |
| Lightning | ✅ | Tree support |
| CrossHatch | ✅ | Direction alternation |
| Honeycomb3D | ✅ | 3D octahedra (needs z_height) |
| HilbertCurve | ✅ | Space-filling curve |
| ArchimedeanChords | ✅ | Spiral from center |
| OctagramSpiral | ✅ | 8-pointed star spiral |
| Adaptive | ⚠️ | Falls back to rectilinear** |
| SupportCubic | ⚠️ | Falls back to rectilinear** |

*FloatingConcentric requires `generate_with_floating_areas()` for proper floating detection; otherwise falls back to regular concentric.

**Adaptive and SupportCubic require an octree built from mesh triangles. Use `AdaptiveInfillGenerator` directly for proper adaptive infill.

## Key Types

### InfillPattern

```rust
pub enum InfillPattern {
    Rectilinear,        // Parallel lines
    Grid,               // Cross-hatch
    Concentric,         // Inward spiral
    FloatingConcentric, // Concentric with floating detection
    Line,               // Single direction lines
    Honeycomb,          // Hexagonal
    Gyroid,             // TPMS surface
    Lightning,          // Tree support
    Adaptive,           // Variable density near surfaces
    SupportCubic,       // Overhang-only densification
    Honeycomb3D,        // 3D truncated octahedron
    CrossHatch,         // Alternating direction with transitions
    HilbertCurve,       // Space-filling curve
    ArchimedeanChords,  // Spiral from center
    OctagramSpiral,     // 8-pointed star spiral
    None,               // No infill
}
```

### InfillConfig

```rust
pub struct InfillConfig {
    pub pattern: InfillPattern,     // Which pattern to use
    pub density: f64,               // 0.0 - 1.0 (0% - 100%)
    pub extrusion_width: f64,       // Line width in mm
    pub angle: f64,                 // Base angle in degrees
    pub angle_increment: f64,       // Rotation per layer
    pub overlap: f64,               // Overlap with perimeters
    pub min_area: f64,              // Minimum region area to fill
    pub connect_infill: bool,       // Connect lines to reduce travel
    pub infill_first: bool,         // Infill before perimeters
    pub z_height: f64,              // Current Z height (for 3D patterns)
    pub layer_height: f64,          // Layer height (for 3D patterns)
}
```

**Note**: `z_height` and `layer_height` are required for 3D patterns (Honeycomb3D, CrossHatch) to generate correct layer-specific patterns.

### InfillResult

```rust
pub struct InfillResult {
    pub paths: Vec<InfillPath>,     // Generated fill paths
    pub total_length_mm: f64,       // Total path length
    pub pattern: InfillPattern,     // Pattern used
    pub density: f64,               // Density used
}
```

### InfillPath

```rust
pub enum InfillPath {
    Line(Polyline),    // Open path
    Loop(Polygon),     // Closed path (concentric)
}
```

## Line Spacing Calculation

The spacing between infill lines is calculated from density:

```rust
fn line_spacing(&self) -> f64 {
    if self.density <= 0.0 {
        return f64::MAX;  // No infill
    }
    if self.density >= 1.0 {
        return self.extrusion_width;  // Solid infill
    }
    // Spacing = width / density
    self.extrusion_width / self.density
}
```

**Example**: 0.4mm width at 20% density → 2mm spacing

## Angle Rotation

Infill angle rotates between layers for better strength distribution:

```rust
fn angle_for_layer(&self, layer_index: usize) -> f64 {
    self.angle + (layer_index as f64 * self.angle_increment)
}
```

**Default**: 45° base, +90° per layer → alternates 45° and 135°

## File Structure

```
slicer/src/infill/
├── AGENTS.md       # This file
├── mod.rs          # Main generator and standard patterns
├── adaptive.rs     # Adaptive cubic infill (octree-based)
├── cross_hatch.rs  # Cross Hatch pattern
└── honeycomb_3d.rs # 3D Honeycomb (truncated octahedron)
```

## Algorithm: Rectilinear Fill

```rust
fn generate_rectilinear(region: &ExPolygon, config: &InfillConfig) -> Vec<Polyline> {
    let bbox = region.bounding_box();
    let spacing = config.line_spacing();
    let angle = config.angle.to_radians();
    
    // 1. Generate parallel lines covering bounding box
    let mut lines = Vec::new();
    let mut y = bbox.min.y;
    while y <= bbox.max.y {
        lines.push(Line::new(
            Point::new(bbox.min.x, y),
            Point::new(bbox.max.x, y),
        ));
        y += spacing;
    }
    
    // 2. Rotate lines by infill angle
    for line in &mut lines {
        line.rotate(angle, bbox.center());
    }
    
    // 3. Clip lines to region (respecting holes)
    let clipped = clip_lines_to_expolygon(&lines, region);
    
    // 4. Optionally connect lines to reduce travel
    if config.connect_infill {
        connect_infill_lines(clipped)
    } else {
        clipped
    }
}
```

## Algorithm: Honeycomb Fill

Honeycomb uses a hexagonal grid pattern:

```rust
// Hexagon dimensions from line spacing
let hex_width = spacing * 2.0;
let hex_height = spacing * sqrt(3.0);

// Offset every other row by half width
let x_offset = if row % 2 == 0 { 0.0 } else { hex_width / 2.0 };
```

## Algorithm: Gyroid Fill

Gyroid is a triply periodic minimal surface defined by:

```
sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
```

For 2D slices at height z, we sample points where the gyroid surface crosses the plane.

## Solid Infill

For top/bottom surfaces, infill uses 100% density with rectilinear pattern:

```rust
pub fn solid() -> Self {
    InfillConfig {
        pattern: InfillPattern::Rectilinear,
        density: 1.0,
        angle: 45.0,
        ..Default::default()
    }
}
```

## Overlap with Perimeters

Infill overlaps slightly with the innermost perimeter for adhesion:

```rust
// Shrink fill region by (perimeter_width - overlap)
let fill_region = shrink_expolygon(
    region,
    perimeter_width - config.overlap
);
```

**Default overlap**: ~0.25× extrusion width

## Dependencies

- `crate::geometry::{ExPolygon, Polygon, Polyline, Point}` - Geometry types
- `crate::clipper` - Polygon clipping operations
- `crate::flow::Flow` - Extrusion calculations

## Related Modules

- `perimeter/` - Generates perimeters that infill fills inside
- `slice/surface.rs` - Identifies which surfaces need solid vs sparse infill
- `gcode/path.rs` - Converts infill paths to ExtrusionPath
- `bridge/` - Bridge infill uses special handling

## Testing Strategy

1. **Unit tests**: Line spacing calculation, angle rotation
2. **Pattern tests**: Each pattern generates expected path shapes
3. **Density tests**: 0%, 50%, 100% produce correct coverage
4. **Clipping tests**: Paths stay within region, respect holes
5. **Parity tests**: Compare output with BambuStudio on test shapes

## Performance Considerations

1. **Spatial indexing**: Use grid for fast line-polygon intersection
2. **Path connection**: Greedy nearest-neighbor for connecting lines
3. **Caching**: Cache pattern generation for repeated densities/angles

## Adaptive Cubic Infill

The `adaptive.rs` module implements octree-based adaptive infill that varies density based on proximity to the model surface.

### Algorithm Overview

1. **Build Octree**: Subdivide 3D space around mesh triangles
2. **Insert Triangles**: Recursively subdivide cells that contain triangles
3. **Generate Lines**: At each Z height, traverse octree and generate lines
4. **Three Directions**: Lines are generated in 3 directions for strength
5. **Connect & Clip**: Connect lines using hooks, clip to boundary

### Key Types

```rust
pub struct AdaptiveInfillConfig {
    pub line_spacing: f64,           // Base line spacing (mm)
    pub extrusion_width: f64,        // Line width (mm)
    pub support_overhangs_only: bool, // Only densify below overhangs
    pub hook_length: f64,            // Hook length for connections (mm)
    pub connect_lines: bool,         // Whether to connect lines
}

pub struct Octree {
    pub root_cube: Option<Box<Cube>>,
    pub origin: Vec3d,
    pub cubes_properties: Vec<CubeProperties>,
}

pub struct AdaptiveInfillGenerator {
    config: AdaptiveInfillConfig,
    octree: Option<Octree>,
}
```

### Coordinate System

The octree is rotated to stand on one of its corners, creating lines in three non-parallel directions for better strength:

```rust
// Rotation angles for octree coordinate system
const OCTREE_ROT: [f64; 3] = [
    5.0 * PI / 4.0,           // X rotation
    215.264_deg.to_radians(), // Y rotation
    PI / 6.0,                 // Z rotation
];
```

### SupportCubic Pattern

`SupportCubic` is a variant that only densifies below internal overhangs, useful for providing support to internal bridge surfaces without adding unnecessary infill elsewhere.

## Cross Hatch Infill

The `cross_hatch.rs` module implements Cross Hatch infill pattern from BambuStudio.

### Algorithm Overview

Cross Hatch alternates line direction by 90° every few layers with smooth transitions:

1. **Repeat Layers**: Simple parallel lines in one direction
2. **Transform Layers**: Zigzag patterns that transition between directions

```
Repeat Layer (horizontal):
────────────────────
────────────────────

Transform Layer (zigzag):
    o---o
   /     \
  /       \
           \       /
            \     /
             o---o
```

### Key Types

```rust
pub struct CrossHatchConfig {
    pub grid_size: f64,        // Distance between lines (mm)
    pub angle: f64,            // Rotation angle (degrees)
    pub repeat_ratio: f64,     // Ratio of repeat to transform layers
    pub density: f64,          // Infill density
    pub connect_lines: bool,   // Connect adjacent lines
    pub min_line_length: f64,  // Filter short segments
}

pub struct CrossHatchResult {
    pub polylines: Vec<Polyline>,
    pub is_transform_layer: bool,
    pub direction: i32,        // -1 = vertical, 1 = horizontal
    pub progress: f64,         // Transform progress (0.0-1.0)
}
```

### Layer Period Calculation

```rust
let trans_layer_size = grid_size * 0.4;     // Transform layer height
let repeat_layer_size = grid_size * repeat_ratio;  // Repeat layer height
let period = trans_layer_size + repeat_layer_size;
```

Reference: BambuStudio `Fill/FillCrossHatch.cpp`

## 3D Honeycomb Infill

The `honeycomb_3d.rs` module implements truncated octahedron tessellation.

### Algorithm Overview

Creates interlocking 3D structure using truncated octahedra (Kelvin cells):

1. **Triangle Wave**: Base waveform for pattern
2. **TrOct Wave**: Composed waveform for truncated octahedron
3. **Grid Generation**: Create grid at specific Z height
4. **Clipping**: Clip to fill region

### Key Types

```rust
pub struct Honeycomb3DConfig {
    pub grid_size: f64,       // Cell size (mm)
    pub extrusion_width: f64, // Line width (mm)
    pub angle: f64,           // Rotation angle (degrees)
}

pub struct Honeycomb3DResult {
    pub polylines: Vec<Polyline>,
    pub z_height: f64,
}
```

### Wave Functions

```rust
// Triangle wave: oscillates between -1 and 1
fn tri_wave(x: f64, period: f64) -> f64

// Truncated octahedron wave: composed of triangle waves
fn troct_wave(x: f64, y: f64, z: f64, period: f64) -> f64
```

Reference: BambuStudio `Fill/Fill3DHoneycomb.cpp` (David Eccles/gringer)

## File Structure

```
slicer/src/infill/
├── AGENTS.md       # This file
├── mod.rs          # Main generator and standard patterns
├── adaptive.rs     # Adaptive cubic infill (octree-based)
├── cross_hatch.rs  # Cross Hatch pattern
├── honeycomb_3d.rs # 3D Honeycomb (truncated octahedron)
└── plan_path.rs    # Space-filling curves (Hilbert, Archimedean, Octagram)
```

## Floating Concentric Infill

Floating Concentric is an enhanced concentric pattern for top surfaces that detects
which parts of the infill are "floating" (not supported by the layer below).

### Algorithm Overview

1. Generate concentric loops by repeatedly shrinking the fill area inward
2. For each loop, detect which vertices/segments intersect with "floating areas"
3. Mark each vertex as floating or supported
4. Optionally split paths at floating/supported transitions
5. Reorder loop starting points to prefer non-floating areas

### Key Types

```rust
pub struct FloatingThickPolyline {
    pub points: Vec<Point>,       // The polyline points
    pub widths: Vec<CoordF>,      // Width at each vertex
    pub is_floating: Vec<bool>,   // Floating flag per vertex
}

pub struct FloatingConcentricConfig {
    pub spacing: CoordF,                    // Line spacing (mm)
    pub min_loop_length: CoordF,            // Minimum loop length to include
    pub loop_clipping: CoordF,              // Distance to clip from loop ends
    pub split_at_transitions: bool,         // Split at floating/supported boundaries
    pub prefer_non_floating_start: bool,    // Start loops at non-floating points
    pub default_width: CoordF,              // Default line width
}

pub struct FloatingConcentricResult {
    pub polylines: Vec<FloatingThickPolyline>,
    pub total_length_mm: CoordF,
    pub floating_fraction: CoordF,  // 0.0-1.0, fraction that is floating
    pub loop_count: usize,
}
```

### Usage Example

```rust
use slicer::{generate_floating_concentric, FloatingConcentricConfig};

// Simple usage
let result = generate_floating_concentric(&fill_area, &floating_areas, 0.4);

// With custom config
let config = FloatingConcentricConfig::new(0.4)
    .with_split_at_transitions(true)
    .with_prefer_non_floating_start(true);
let result = generate_floating_concentric_with_config(&fill_area, &floating_areas, config);

// Access floating information
for polyline in &result.polylines {
    for (i, &is_floating) in polyline.is_floating.iter().enumerate() {
        if is_floating {
            // Adjust extrusion for floating section
        }
    }
}
```

### libslic3r Reference

- `src/libslic3r/Fill/FillFloatingConcentric.cpp`
- `src/libslic3r/Fill/FillFloatingConcentric.hpp`

Key differences from C++ implementation:
- Uses point sampling for floating detection instead of Clipper_Z
- Simplified path merging without complex Z-value tracking
- Direct integration with existing concentric generation

## Future Enhancements

| Feature | Status | libslic3r Reference |
|---------|--------|---------------------|
| Rectilinear | ✅ Done | `FillRectilinear.cpp` |
| Grid | ✅ Done | `FillRectilinear.cpp` |
| Honeycomb | ✅ Done | `FillHoneycomb.cpp` |
| Gyroid | ✅ Done | `FillGyroid.cpp` |
| Concentric | ✅ Done | `FillConcentric.cpp` |
| Lightning | ✅ Done | `FillLightning.cpp` |
| Adaptive Cubic | ✅ Done | `FillAdaptive.cpp` |
| Support Cubic | ✅ Done | `FillAdaptive.cpp` |
| 3D Honeycomb | ✅ Done | `Fill3DHoneycomb.cpp` |
| Cross Hatch | ✅ Done | `FillCrossHatch.cpp` |
| Hilbert Curve | ✅ Done | `FillPlanePath.cpp` |
| Archimedean Chords | ✅ Done | `FillPlanePath.cpp` |
| Octagram Spiral | ✅ Done | `FillPlanePath.cpp` |
| Floating Concentric | 📋 Planned | `FillFloatingConcentric.cpp` |

## Space-Filling Curve Patterns

The `plan_path.rs` module implements mathematical space-filling curves from the Math::PlanePath library.

### Hilbert Curve

A continuous fractal curve that fills a square region. The curve visits every point in a 2^n × 2^n grid exactly once.

```
┌──┐┌──┐
│  └┘  │
└┐┌──┐┌┘
┌┘└──┘└┐
│  ┌┐  │
└──┘└──┘
```

**Algorithm**: Uses a state machine with 4 states (plain, transpose, rot180, rot180+transpose) to convert linear index to 2D coordinates.

### Archimedean Chords

A spiral pattern from center outward following: r = a + b*θ

```
    ╭───╮
  ╭─╯   ╰─╮
 │    ●    │
  ╰─╮   ╭─╯
    ╰───╯
```

**Use case**: Good for circular parts, interesting visual effect.

### Octagram Spiral

An 8-pointed star pattern that spirals outward.

```
    *
   /|\
  * * *
   \|/
    *
```

**Use case**: Decorative infill, unique visual appearance.

### Key Types

```rust
pub struct PlanPathConfig {
    pub spacing: f64,           // Line spacing (mm)
    pub density: f64,           // Infill density (0.0-1.0)
    pub resolution: f64,        // Curve discretization (mm)
    pub align_to_object: bool,  // Align across layers
    pub connect_lines: bool,    // Connect polylines
}

pub enum PlanPathPattern {
    HilbertCurve,
    ArchimedeanChords,
    OctagramSpiral,
}

pub struct PlanPathGenerator {
    config: PlanPathConfig,
    pattern: PlanPathPattern,
}
```

Reference: BambuStudio `Fill/FillPlanePath.cpp` (Math::PlanePath library)