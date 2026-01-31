# Bridge Module

## Purpose

The `bridge` module detects and handles bridging regions - areas where the printer must extrude filament across open gaps without support from below. Proper bridge detection and handling is critical for print quality, as bridges require special extrusion parameters (flow, speed, cooling) to produce clean unsupported spans.

## What This Module Does

1. **Bridge Detection**: Identify regions that span over gaps or voids
2. **Direction Optimization**: Calculate optimal bridging direction for shortest spans
3. **Anchor Analysis**: Find solid anchor regions on either side of a bridge
4. **Bridge Infill**: Generate infill paths optimized for bridging
5. **Flow/Speed Adjustment**: Provide parameters for bridge-specific extrusion

## libslic3r Reference

| Rust File | C++ File(s) | Description |
|-----------|-------------|-------------|
| `mod.rs` | `BridgeDetector.hpp/cpp` | Main bridge detection |
| | `InternalBridgeDetector.hpp/cpp` | Internal bridge detection |

### Key C++ Classes

- **`BridgeDetector`**: Detects bridges and computes optimal direction
- **`InternalBridgeDetector`**: Detects internal bridges (infill over sparse infill)
- Uses `Surface` with `stBottomBridge` type

## Bridge Detection

### What is a Bridge?

A bridge occurs when material must be extruded across a gap:

```
       Previous layer
           ↓
    ████████████████
    ██            ██      ← Gap (no support below)
    ██            ██
    ██████████████████    ← Current layer needs bridging
         ↑
    Bridge region
```

### Detection Algorithm

```rust
fn detect_bridges(
    current_layer: &[ExPolygon],
    previous_layer: &[ExPolygon],
) -> Vec<Bridge> {
    // Find bottom surfaces (areas without support from below)
    let supported = intersection(current_layer, previous_layer);
    let unsupported = difference(current_layer, &supported);
    
    // Filter to areas that span gaps (not just overhangs)
    let mut bridges = Vec::new();
    for region in unsupported {
        if is_bridge(&region, previous_layer) {
            bridges.push(Bridge::new(region));
        }
    }
    
    bridges
}

fn is_bridge(region: &ExPolygon, below: &[ExPolygon]) -> bool {
    // A bridge has anchors on opposite sides
    // (not just an overhang extending from one side)
    let anchors = find_anchors(region, below);
    anchors.len() >= 2 && anchors_on_opposite_sides(&anchors)
}
```

## Key Types

### Bridge

```rust
pub struct Bridge {
    pub region: ExPolygon,           // The bridging region
    pub direction: f64,              // Optimal bridge angle (radians)
    pub length: f64,                 // Bridge span length (mm)
    pub anchors: Vec<Polygon>,       // Anchor regions
}
```

### BridgeConfig

```rust
pub struct BridgeConfig {
    pub min_area: f64,               // Minimum area to consider (mm²)
    pub max_bridge_length: f64,      // Maximum span length (mm)
    pub flow_multiplier: f64,        // Flow ratio (typically 1.0-1.2)
    pub speed_multiplier: f64,       // Speed ratio (typically 0.5-0.8)
    pub fan_boost: bool,             // Increase cooling for bridges
}
```

### BridgeDetector

```rust
pub struct BridgeDetector {
    config: BridgeConfig,
}

impl BridgeDetector {
    pub fn detect(&self, current: &[ExPolygon], below: &[ExPolygon]) -> Vec<Bridge>;
    pub fn compute_direction(&self, bridge: &Bridge) -> f64;
    pub fn generate_infill(&self, bridge: &Bridge) -> Vec<Polyline>;
}
```

## Bridge Direction Optimization

The optimal bridge direction minimizes span length:

```
Bad direction (long span):          Good direction (short span):
        ←──────────────→                     ↑
    ████                ████             ████│████
    ████   ══════════   ████             ████│████
    ████                ████             ████│████
                                             ↓
    Span = full width                   Span = narrow gap
```

### Algorithm

```rust
fn compute_optimal_direction(bridge: &Bridge) -> f64 {
    let mut best_angle = 0.0;
    let mut min_length = f64::MAX;
    
    // Try angles from 0° to 180° in small increments
    for angle_deg in (0..180).step_by(5) {
        let angle = angle_deg as f64 * PI / 180.0;
        
        // Compute maximum span length at this angle
        let length = compute_span_at_angle(bridge, angle);
        
        if length < min_length {
            min_length = length;
            best_angle = angle;
        }
    }
    
    best_angle
}

fn compute_span_at_angle(bridge: &Bridge, angle: f64) -> f64 {
    // Cast rays across the bridge at the given angle
    // Return the maximum ray length
    let rays = cast_parallel_rays(&bridge.region, angle);
    rays.iter().map(|r| r.length()).max().unwrap_or(0.0)
}
```

## Anchor Detection

Anchors are the solid regions on either side of a bridge:

```rust
fn find_anchors(bridge: &ExPolygon, below: &[ExPolygon]) -> Vec<Polygon> {
    // Grow the bridge region slightly
    let expanded = grow(bridge, anchor_search_distance);
    
    // Intersect with layer below to find anchor areas
    let anchors = intersection(&expanded, below);
    
    // Filter to significant anchor areas
    anchors.into_iter()
        .filter(|a| a.area() > min_anchor_area)
        .collect()
}
```

## Bridge Infill

Bridge infill uses straight lines perpendicular to the gap:

```rust
fn generate_bridge_infill(bridge: &Bridge) -> Vec<Polyline> {
    let direction = bridge.direction;
    let spacing = bridge_extrusion_width;  // Usually 100% density
    
    // Generate parallel lines at bridge angle
    let lines = generate_parallel_lines(
        &bridge.region,
        direction,
        spacing,
    );
    
    // Clip to bridge region
    clip_to_region(&lines, &bridge.region)
}
```

## Bridge Extrusion Parameters

Bridges require special extrusion settings:

| Parameter | Normal | Bridge | Why |
|-----------|--------|--------|-----|
| Flow | 100% | 100-120% | More material for sag |
| Speed | Normal | 50-80% | Slower for adhesion |
| Cooling | Normal | Maximum | Quick solidification |
| Width | Normal | Same or wider | Better anchoring |

### Flow Calculation

```rust
// Bridge flow from libslic3r/Flow.cpp
fn bridge_flow(nozzle_diameter: f64) -> Flow {
    // Bridge uses circular cross-section
    // (unsupported filament naturally forms round thread)
    Flow::bridging_flow(nozzle_diameter, nozzle_diameter)
}
```

## Internal Bridges

Internal bridges occur when infill must span over sparse infill below:

```
    Top solid layer
         ↓
    ████████████████
      ╎╎╎╎╎╎╎╎╎╎╎╎      ← Sparse infill (gaps)
    ████████████████    ← Internal bridge needed
      ╎╎╎╎╎╎╎╎╎╎╎╎
    ████████████████
```

These are detected similarly but may use different parameters.

## File Structure

```
slicer/src/bridge/
├── AGENTS.md    # This file
└── mod.rs       # Bridge detection and handling
```

## Integration with Pipeline

```
slice/ → detect layers
    │
    ▼
bridge/ → detect bridge regions
    │
    ▼
perimeter/ → mark bridge perimeters
    │
    ▼
infill/ → generate bridge infill
    │
    ▼
gcode/ → apply bridge flow/speed
```

## Dependencies

- `crate::geometry::{ExPolygon, Polygon, Polyline, Point}` - Geometry types
- `crate::clipper::{intersection, difference, grow}` - Boolean operations
- `crate::flow::Flow` - Bridge flow calculations
- `crate::slice::Layer` - Layer data

## Related Modules

- `slice/surface.rs` - Surface types include `BottomBridge`
- `flow/` - Bridge flow calculations (circular cross-section)
- `gcode/path.rs` - `ExtrusionRole::BridgeInfill` role
- `infill/` - Bridge infill generation
- `gcode/cooling.rs` - Fan boost for bridges

## Testing Strategy

1. **Unit tests**: Bridge detection on known shapes
2. **Direction tests**: Verify optimal direction calculation
3. **Anchor tests**: Correct anchor identification
4. **Flow tests**: Bridge flow uses circular formula
5. **Parity tests**: Compare with BambuStudio bridge detection

## Common Issues

### False Positives
- Small overhangs detected as bridges
- Solution: Minimum area threshold

### Incorrect Direction
- Suboptimal direction chosen
- Solution: Finer angle search, anchor-based direction

### Poor Bridge Quality
- Sagging or drooping
- Solution: Adjust flow, speed, cooling parameters

## Bridge Quality Tips

1. **Slow down**: Reduces sag from momentum
2. **More cooling**: Solidifies quickly
3. **Good anchors**: Strong attachment at ends
4. **Short spans**: Keep bridges under 50mm if possible
5. **No retraction**: Continuous extrusion over bridge

## Future Enhancements

| Feature | Status | libslic3r Reference |
|---------|--------|---------------------|
| Bridge detection | ✅ Done | `BridgeDetector.cpp` |
| Direction optimization | ✅ Done | `BridgeDetector.cpp` |
| Bridge infill | ✅ Done | `BridgeDetector.cpp` |
| Bridge flow | ✅ Done | `Flow.cpp` |
| Internal bridges | 🔄 Partial | `InternalBridgeDetector.cpp` |
| Thick bridges | 📋 Planned | Multiple bridge layers |
| Bridge fan control | ✅ Done | `CoolingBuffer.cpp` |