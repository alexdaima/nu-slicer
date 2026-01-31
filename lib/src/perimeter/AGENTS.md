# Perimeter Module

## Purpose

The `perimeter` module generates the outer shells (perimeters) of each layer. Perimeters define the visible surface quality of printed objects and provide structural integrity. This module supports both classic fixed-width perimeters and Arachne variable-width perimeters.

## What This Module Does

1. **Perimeter Generation**: Create inward-offset shells from layer boundaries
2. **Classic Mode**: Fixed-width perimeters using simple polygon offsetting
3. **Arachne Mode**: Variable-width perimeters that adapt to local geometry
4. **Thin Wall Detection**: Identify areas too narrow for full perimeters
5. **Gap Fill**: Generate paths for gaps between perimeters
6. **Infill Area Calculation**: Compute the region remaining for infill after perimeters

## libslic3r Reference

| Rust File | C++ File(s) | Description |
|-----------|-------------|-------------|
| `mod.rs` | `PerimeterGenerator.hpp/cpp` | Main perimeter generation |
| `arachne/mod.rs` | `Arachne/WallToolPaths.cpp` | Arachne wall generation |
| `arachne/beading.rs` | `Arachne/BeadingStrategy.cpp` | Bead width calculation |
| `arachne/junction.rs` | `Arachne/ExtrusionJunction.hpp` | Junction points |
| `arachne/line.rs` | `Arachne/ExtrusionLine.hpp` | Variable-width lines |

### Key C++ Classes

- **`PerimeterGenerator`**: Main entry point for perimeter generation
- **`Arachne::WallToolPaths`**: Variable-width wall computation
- **`Arachne::BeadingStrategy`**: Calculates optimal bead widths
- **`Arachne::ExtrusionLine`**: Line with variable width along length
- **`Arachne::ExtrusionJunction`**: Point with associated width

## Perimeter Generation Modes

### Classic Mode

Traditional fixed-width perimeters created by polygon offsetting:

```
Original slice    After 3 perimeters
┌────────────┐    ┌────────────┐
│            │    │ ┌────────┐ │
│            │    │ │ ┌────┐ │ │
│            │    │ │ │    │ │ │  ← Infill area
│            │    │ │ └────┘ │ │
│            │    │ └────────┘ │
└────────────┘    └────────────┘
     ↑                  ↑
  Outer            3 shells
 contour           inward
```

**Algorithm**:
1. Start with slice ExPolygon
2. Offset inward by perimeter width
3. Repeat for number of perimeters
4. Remaining area becomes infill region

### Arachne Mode (Variable Width)

Adaptive perimeters that vary width based on local geometry:

```
      Narrow section
         ↓↓↓
┌────────╲╱────────┐
│ ████████████████ │  ← Outer perimeter (full width)
│ ████╲    ╱████ │  ← Inner perimeter (reduced width)
│ ████ ╲  ╱ ████ │  ← Tapered to fit
│ ████  ╲╱  ████ │
└─────────────────┘
```

**Benefits**:
- Better thin wall handling
- No gaps in narrow sections  
- Smoother surface finish
- Stronger parts

## Key Types

### PerimeterConfig

```rust
pub struct PerimeterConfig {
    pub perimeter_count: u32,           // Number of shells
    pub external_perimeter_width: f64,  // Outer perimeter width (mm)
    pub perimeter_width: f64,           // Inner perimeter width (mm)
    pub thin_walls: bool,               // Generate thin wall fills
    pub gap_fill: bool,                 // Fill gaps between perimeters
    pub external_perimeters_first: bool, // Print order
}
```

### PerimeterResult

```rust
pub struct PerimeterResult {
    pub perimeters: Vec<PerimeterLoop>,  // Generated perimeter loops
    pub thin_fills: Vec<ExtrusionPath>,  // Thin wall fills
    pub gap_fills: Vec<ExtrusionPath>,   // Gap fill paths
    pub infill_area: Vec<ExPolygon>,     // Area for infill
}
```

### PerimeterLoop

```rust
pub struct PerimeterLoop {
    pub polygon: Polygon,       // The perimeter path
    pub is_external: bool,      // Is outermost perimeter
    pub depth: u32,             // 0 = external, 1 = first inner, etc.
    pub width: f64,             // Extrusion width
}
```

## Arachne Submodule

The `arachne/` submodule implements variable-width perimeters based on the Arachne algorithm from Ultimaker Cura.

### BeadingStrategy

Determines how to distribute wall widths:

```rust
pub struct BeadingResult {
    pub bead_widths: Vec<f64>,      // Width of each bead
    pub toolpath_locations: Vec<f64>, // Center positions
    pub total_width: f64,            // Total filled width
}
```

### ExtrusionJunction

A point with an associated extrusion width:

```rust
pub struct ExtrusionJunction {
    pub position: Point,
    pub width: f64,        // Width at this point
    pub perimeter_index: u32,
}
```

### ExtrusionLine

A polyline with variable width (width can change along the line):

```rust
pub struct ExtrusionLine {
    pub junctions: Vec<ExtrusionJunction>,
    pub is_closed: bool,
    pub inset_index: u32,
    pub is_odd: bool,     // Part of an odd-count wall set
}
```

## File Structure

```
slicer/src/perimeter/
├── AGENTS.md           # This file
├── mod.rs              # Main perimeter generation
└── arachne/            # Variable-width perimeters
    ├── AGENTS.md       # Arachne documentation
    ├── mod.rs          # Arachne entry point
    ├── beading.rs      # Bead width calculation
    ├── junction.rs     # ExtrusionJunction type
    └── line.rs         # ExtrusionLine type
```

## Classic Perimeter Algorithm

```rust
fn generate_classic_perimeters(
    slice: &ExPolygon,
    config: &PerimeterConfig,
) -> PerimeterResult {
    let mut perimeters = Vec::new();
    let mut current = slice.clone();
    
    for i in 0..config.perimeter_count {
        let width = if i == 0 {
            config.external_perimeter_width
        } else {
            config.perimeter_width
        };
        
        // Offset inward by half width (to centerline)
        let offset = if i == 0 {
            width / 2.0
        } else {
            width
        };
        
        let shells = shrink_expolygon(&current, offset);
        
        for shell in &shells {
            perimeters.push(PerimeterLoop {
                polygon: shell.contour.clone(),
                is_external: i == 0,
                depth: i,
                width,
            });
            
            // Add hole perimeters
            for hole in &shell.holes {
                perimeters.push(PerimeterLoop {
                    polygon: hole.clone(),
                    is_external: i == 0,
                    depth: i,
                    width,
                });
            }
        }
        
        current = shells;
    }
    
    // Remaining area is for infill
    let infill_area = shrink_expolygon(&current, config.perimeter_width / 2.0);
    
    PerimeterResult {
        perimeters,
        thin_fills: Vec::new(),
        gap_fills: Vec::new(),
        infill_area,
    }
}
```

## Perimeter Spacing

Perimeters are spaced to ensure proper bonding:

```rust
// External perimeter: offset by half width from slice edge
// Inner perimeters: offset by full width from previous
//
// ─────────────────  ← Slice edge
//      ████████      ← External perimeter (width/2 from edge)
//        ████        ← Inner perimeter (width from external)
//          █         ← Another inner (width from previous)
```

## Thin Wall Detection

Areas too narrow for full perimeters are detected and filled specially:

```rust
fn detect_thin_walls(
    slice: &ExPolygon,
    perimeters: &[PerimeterLoop],
    min_width: f64,
) -> Vec<ExPolygon> {
    // Find areas not covered by perimeters
    let covered = union(&perimeters.iter().map(|p| p.polygon.clone()).collect());
    let uncovered = difference(slice, &covered);
    
    // Filter to areas narrower than min_width
    uncovered.into_iter()
        .filter(|region| region.min_width() < min_width)
        .collect()
}
```

## Gap Fill

Gaps between perimeters are filled with thin extrusions:

```
┌──────────────────┐
│ ████        ████ │
│ ████  ----  ████ │  ← Gap fill paths (----)
│ ████        ████ │
└──────────────────┘
```

## External Perimeters First

When `external_perimeters_first` is enabled:
- Outer perimeter prints first (better surface quality)
- Inner perimeters print after (may show through)

When disabled (default):
- Inner perimeters print first
- Outer perimeter prints last (better dimensional accuracy)

## Dependencies

- `crate::geometry::{ExPolygon, Polygon, Point}` - Geometry types
- `crate::clipper::{shrink, grow, difference}` - Polygon operations
- `crate::flow::Flow` - Extrusion calculations

## Related Modules

- `geometry/` - Provides ExPolygon, Polygon types
- `clipper/` - Polygon offset operations
- `infill/` - Fills the area inside perimeters
- `gcode/path.rs` - Converts perimeters to ExtrusionPath
- `flow/` - Calculates extrusion parameters

## Testing Strategy

1. **Unit tests**: Offset calculations, loop ordering
2. **Shape tests**: Square, circle, complex shapes
3. **Thin wall tests**: Narrow features handled correctly
4. **Arachne tests**: Variable width produces valid paths
5. **Parity tests**: Compare with BambuStudio output

## Performance Considerations

1. **Clipper efficiency**: Use Clipper2 for fast offsetting
2. **Loop ordering**: Sort loops for minimal travel
3. **Parallel processing**: Independent layers can be processed in parallel

## Arachne vs Classic

| Feature | Classic | Arachne |
|---------|---------|---------|
| Wall width | Fixed | Variable |
| Thin walls | Gaps possible | Filled properly |
| Speed | Faster | Slower |
| Quality | Good | Better |
| Memory | Less | More |

**Recommendation**: Use Arachne for quality, Classic for speed.

## Future Enhancements

| Feature | Status | libslic3r Reference |
|---------|--------|---------------------|
| Classic perimeters | ✅ Done | `PerimeterGenerator.cpp` |
| Arachne variable-width | ✅ Done | `Arachne/` directory |
| Thin wall detection | ✅ Done | `PerimeterGenerator.cpp` |
| Gap fill | ✅ Done | `PerimeterGenerator.cpp` |
| Seam position control | 🔄 Partial | `GCode/SeamPlacer.cpp` |
| Fuzzy skin | 📋 Planned | `FuzzySkin.cpp` |
| Extra perimeters on overhangs | 📋 Planned | `PerimeterGenerator.cpp` |