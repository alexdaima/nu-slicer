# Adhesion Module

## Purpose

The `adhesion` module generates bed adhesion features - brim, skirt, and raft structures that help the first layer stick to the build plate and provide a stable foundation for the print. These features are critical for preventing warping and ensuring print success.

## What This Module Does

1. **Brim Generation**: Create outward-extending loops from the first layer perimeter
2. **Skirt Generation**: Generate loops around (but not touching) the print
3. **Raft Configuration**: Define raft layer parameters (generation pending)
4. **Convex Hull Calculation**: Compute print footprint for skirt/brim placement

## libslic3r Reference

| Rust Type | C++ File(s) | Description |
|-----------|-------------|-------------|
| `BrimConfig` | `Brim.hpp/cpp` | Brim configuration |
| `BrimGenerator` | `Brim.cpp` | Brim generation algorithm |
| `SkirtConfig` | `Print.hpp`, `PrintConfig.hpp` | Skirt settings |
| `SkirtGenerator` | `Print.cpp` (`_make_skirt()`) | Skirt generation |
| `RaftConfig` | `PrintConfig.hpp` | Raft layer settings |

### Key C++ Functions

- **`make_brim()`** (`Brim.cpp`): Main brim generation
- **`make_brim_ears()`**: Brim with ear shapes at corners
- **`Print::_make_skirt()`**: Skirt loop generation
- **Raft**: Integrated into support material generation

## Adhesion Types

### Brim

Outward loops attached to the first layer perimeter:

```
              Brim loops
                 ↓↓↓
    ─────────────────────────
    │   ┌─────────────┐   │
    │   │             │   │  ← Part first layer
    │   │   Part      │   │
    │   │             │   │
    │   └─────────────┘   │
    ─────────────────────────
```

**Purpose**: 
- Increases first layer surface area for adhesion
- Prevents corner lifting and warping
- Easy to remove after printing

### Skirt

Loops around the print that don't touch it:

```
    ┌───────────────────────┐
    │                       │  ← Skirt loops
    │   ┌─────────────┐     │
    │   │             │     │
    │   │   Part      │     │     Gap between
    │   │             │     │     skirt and part
    │   └─────────────┘     │
    │                       │
    └───────────────────────┘
```

**Purpose**:
- Primes the extruder before printing
- Shows print footprint on bed
- Detects leveling issues early
- No adhesion benefit (doesn't touch part)

### Raft

Multi-layer platform under the print:

```
    ████████████████████████  ← Part first layer
    ════════════════════════  ← Raft interface (dense)
    ════════════════════════  ← Raft interface
    ╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎╎  ← Raft base (sparse)
    ════════════════════════  ← Raft first layer (on bed)
    ────────────────────────  ← Build plate
```

**Purpose**:
- Maximum adhesion for difficult materials (ABS, Nylon)
- Compensates for uneven bed
- Creates smooth bottom surface
- Required for some support configurations

## Key Types

### BrimConfig

```rust
pub struct BrimConfig {
    pub width: f64,               // Total brim width (mm)
    pub line_spacing: f64,        // Spacing between brim lines
    pub loops: Option<u32>,       // Number of loops (alternative to width)
    pub gap: f64,                 // Gap between brim and part
    pub brim_type: BrimType,      // Outer, Inner, or Both
    pub enabled: bool,
    pub extrusion_width: f64,     // Brim line width
    pub min_area: f64,            // Minimum part area for brim
    pub smart_brim: bool,         // Skip brim on large flat areas
}
```

### BrimType

```rust
pub enum BrimType {
    Outer,   // Brim on outside of part only
    Inner,   // Brim in holes only
    Both,    // Brim on both sides
}
```

### SkirtConfig

```rust
pub struct SkirtConfig {
    pub loops: u32,               // Number of skirt loops
    pub distance: f64,            // Distance from part (mm)
    pub min_length: f64,          // Minimum total extrusion length
    pub height: u32,              // Number of layers for skirt
    pub enabled: bool,
    pub extrusion_width: f64,
    pub draft_shield: bool,       // Extend skirt full height
}
```

### RaftConfig

```rust
pub struct RaftConfig {
    pub layers: u32,              // Total raft layers
    pub contact_distance: f64,    // Gap to part
    pub expansion: f64,           // How far raft extends beyond part
    pub first_layer_density: f64, // Bottom raft layer density
    pub interface_layers: u32,    // Dense interface layer count
}
```

### BrimResult / SkirtResult

```rust
pub struct BrimResult {
    pub loops: Vec<Polygon>,      // Generated brim loops
    pub total_length: f64,        // Total extrusion length
}

pub struct SkirtResult {
    pub loops: Vec<Polygon>,      // Generated skirt loops
    pub height_layers: u32,       // How many layers
}
```

## Generation Algorithms

### Brim Generation

```rust
fn generate_brim(first_layer_slices: &[ExPolygon], config: &BrimConfig) -> BrimResult {
    if !config.enabled {
        return BrimResult::empty();
    }
    
    // 1. Union all first layer contours
    let unified = union_ex(first_layer_slices);
    
    // 2. Generate outward offsets
    let mut loops = Vec::new();
    let mut current = unified;
    let num_loops = config.loops.unwrap_or_else(|| 
        (config.width / config.line_spacing).ceil() as u32
    );
    
    for i in 0..num_loops {
        let offset = if i == 0 {
            config.gap + config.extrusion_width / 2.0
        } else {
            config.line_spacing
        };
        
        current = grow(&current, offset);
        
        for expolygon in &current {
            loops.push(expolygon.contour.clone());
        }
    }
    
    BrimResult { 
        loops,
        total_length: calculate_total_length(&loops),
    }
}
```

### Skirt Generation

```rust
fn generate_skirt(first_layer_slices: &[ExPolygon], config: &SkirtConfig) -> SkirtResult {
    if !config.enabled || first_layer_slices.is_empty() {
        return SkirtResult::empty();
    }
    
    // 1. Compute convex hull of all first layer geometry
    let all_points: Vec<Point> = first_layer_slices.iter()
        .flat_map(|ex| ex.contour.points())
        .cloned()
        .collect();
    let hull = convex_hull(&all_points);
    
    // 2. Generate offset loops
    let mut loops = Vec::new();
    let mut current = vec![hull];
    
    for i in 0..config.loops {
        let offset = if i == 0 {
            config.distance + config.extrusion_width / 2.0
        } else {
            config.extrusion_width
        };
        
        current = grow(&current, offset);
        loops.extend(current.clone());
    }
    
    // 3. Ensure minimum length
    while calculate_total_length(&loops) < config.min_length {
        current = grow(&current, config.extrusion_width);
        loops.extend(current.clone());
    }
    
    SkirtResult {
        loops,
        height_layers: config.height,
    }
}
```

## Convex Hull

Used to compute the print footprint for skirt generation:

```rust
fn convex_hull(points: &[Point]) -> Polygon {
    // Graham scan or Andrew's monotone chain algorithm
    // Returns the convex hull as a polygon
}
```

## File Structure

```
slicer/src/adhesion/
├── AGENTS.md    # This file
└── mod.rs       # All adhesion implementations
```

## Integration with Pipeline

```
mesh/ → slice first layer
    │
    ▼
adhesion/ → generate brim/skirt/raft
    │
    ▼
gcode/ → output as first layer features
```

Adhesion features are generated after slicing but before regular toolpath generation. They print first on layer 0.

## Extrusion Role

Brim and skirt use `ExtrusionRole::Skirt`:

```rust
impl BrimResult {
    pub fn to_extrusion_paths(&self, config: &BrimConfig) -> Vec<ExtrusionPath> {
        self.loops.iter()
            .map(|polygon| {
                ExtrusionPath::from_polygon(polygon, ExtrusionRole::Skirt)
                    .with_width(config.extrusion_width)
            })
            .collect()
    }
}
```

## Dependencies

- `crate::geometry::{ExPolygon, Polygon, Point}` - Geometry types
- `crate::clipper::{union_ex, grow}` - Boolean operations
- `crate::gcode::ExtrusionPath` - Toolpath conversion
- `crate::flow::Flow` - Extrusion calculations

## Related Modules

- `slice/` - Provides first layer geometry
- `clipper/` - Union and offset operations
- `gcode/path.rs` - ExtrusionPath and ExtrusionRole
- `pipeline/` - Orchestrates adhesion generation

## Testing Strategy

1. **Unit tests**: Config defaults, validation
2. **Generation tests**: Correct number of loops
3. **Geometry tests**: Brim attached to part, skirt separated
4. **Length tests**: Minimum skirt length satisfied
5. **Parity tests**: Compare with BambuStudio output

## When to Use Each Type

| Feature | Use Case | Adhesion | Easy Removal |
|---------|----------|----------|--------------|
| None | PLA on textured PEI | N/A | N/A |
| Skirt | Most prints | None | N/A |
| Brim | Tall/thin parts, ABS | Good | Easy |
| Raft | Very warpy materials, uneven bed | Excellent | Moderate |

## Smart Brim

When `smart_brim` is enabled:
- Skip brim on areas with good bed contact
- Only add brim where needed (corners, thin sections)
- Reduces material waste and cleanup time

## Draft Shield

When `draft_shield` is enabled on skirt:
- Skirt extends to full print height
- Creates protective wall around print
- Blocks drafts that cause warping
- Useful for ABS and other temperature-sensitive materials

## Future Enhancements

| Feature | Status | libslic3r Reference |
|---------|--------|---------------------|
| Brim generation | ✅ Done | `Brim.cpp` |
| Skirt generation | ✅ Done | `Print.cpp` |
| Brim types (outer/inner/both) | ✅ Done | `Brim.cpp` |
| Raft config | ✅ Done | `PrintConfig.hpp` |
| Raft generation | 📋 Planned | `SupportMaterial.cpp` |
| Smart brim | 📋 Planned | `Brim.cpp` |
| Brim ears | 📋 Planned | `Brim.cpp` |
| Draft shield | 📋 Planned | `Print.cpp` |
| Mouse ears | 📋 Planned | Corner reinforcement |