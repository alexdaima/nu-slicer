# Flow Module

## Purpose

The `flow` module calculates extrusion flow parameters - the fundamental math that converts desired extrusion dimensions (width, height) into actual material flow rates (mm³/mm of travel). This is **critical** for print quality: incorrect flow calculations result in under-extrusion or over-extrusion.

## What This Module Does

1. **Cross-Section Area Calculation (`mm3_per_mm`)**: Computes the volume of material extruded per millimeter of travel, accounting for the actual physical shape of extruded plastic (a rectangle with semicircular ends, not a simple rectangle).

2. **Extrusion Spacing**: Calculates the optimal centerline-to-centerline distance between adjacent extrusion lines to achieve proper bonding without over-overlap.

3. **Bridge Flow**: Special handling for bridging extrusions which use circular cross-sections (unsupported filament forms a round thread).

4. **Auto Width Calculation**: Derives sensible default extrusion widths based on nozzle diameter and extrusion role.

5. **Flow Ratio Adjustments**: Supports modifying flow while maintaining proper spacing (e.g., for first layer squish, bridge flow ratios).

## libslic3r Reference

This module is a **direct port** of:

| Rust | C++ (BambuStudio/libslic3r) |
|------|----------------------------|
| `flow/mod.rs` | `src/libslic3r/Flow.hpp`, `src/libslic3r/Flow.cpp` |

### Key Functions Mapped

| Rust Function | C++ Function | Description |
|--------------|--------------|-------------|
| `Flow::mm3_per_mm()` | `Flow::mm3_per_mm()` | Cross-section area (mm³/mm) |
| `Flow::spacing()` | `Flow::spacing()` | Line spacing for overlap |
| `Flow::rounded_rectangle_extrusion_spacing()` | `Flow::rounded_rectangle_extrusion_spacing()` | Spacing formula |
| `Flow::rounded_rectangle_extrusion_width_from_spacing()` | `Flow::rounded_rectangle_extrusion_width_from_spacing()` | Inverse of spacing |
| `Flow::bridge_extrusion_spacing()` | `Flow::bridge_extrusion_spacing()` | Bridge thread spacing |
| `Flow::bridging_flow()` | `Flow::bridging_flow()` | Create bridge flow |
| `Flow::auto_extrusion_width()` | `Flow::auto_extrusion_width()` | Default width by role |
| `Flow::with_width()` | `Flow::with_width()` | Clone with new width |
| `Flow::with_height()` | `Flow::with_height()` | Clone with new height |
| `Flow::with_spacing()` | `Flow::with_spacing()` | Adjust for new spacing |
| `Flow::with_cross_section()` | `Flow::with_cross_section()` | Adjust for target area |
| `Flow::with_flow_ratio()` | `Flow::with_flow_ratio()` | Scale flow by ratio |

### Critical Formula: `mm3_per_mm()`

The core formula that **must match exactly**:

```
For normal extrusions (rounded rectangle cross-section):
  area = height × (width - height × (1 - π/4))
       ≈ height × (width - 0.2146 × height)

For bridges (circular cross-section):
  area = π × (width/2)²
       = width² × π/4
```

This models extruded plastic as a rectangle with semicircular ends - the actual physical shape when plastic is deposited.

**Common mistake**: Using `width × height` (simple rectangle) gives ~10-15% error in extrusion calculations.

## Usage in Pipeline

The `Flow` struct should be used wherever extrusion amounts are calculated:

1. **Perimeter generation** - Determines line spacing and overlap
2. **Infill generation** - Calculates fill line spacing
3. **G-code generation** - Converts path length to E-axis values
4. **Support generation** - Support line flow rates
5. **Bridge detection** - Bridge-specific flow parameters

## FlowRole Enum

Extrusion roles affect default width calculations:

| Role | Default Width | Notes |
|------|--------------|-------|
| `ExternalPerimeter` | 1.125 × nozzle | Outer visible surface |
| `Perimeter` | 1.125 × nozzle | Inner perimeters |
| `Infill` | 1.125 × nozzle | Sparse infill |
| `SolidInfill` | 1.125 × nozzle | Top/bottom solid fill |
| `TopSolidInfill` | 1.0 × nozzle | Top surface (finer) |
| `SupportMaterial` | 1.0 × nozzle | Support structures |
| `SupportMaterialInterface` | 1.0 × nozzle | Support interface |
| `SupportTransition` | bridge flow | Tree support transitions |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BRIDGE_EXTRA_SPACING` | 0.05 mm | Extra gap between bridge threads |

## Error Handling

The module defines specific error types matching libslic3r:

- `FlowError::NegativeSpacing` - Width/height combination produces invalid spacing
- `FlowError::NegativeFlow` - Configuration produces negative extrusion (impossible)
- `FlowError::InvalidArgument` - Invalid parameters (e.g., negative dimensions)

## Testing Strategy

Tests should verify:

1. **Exact numerical parity** with libslic3r for representative inputs
2. **Edge cases**: very thin layers, wide extrusions, bridges
3. **Round-trip accuracy**: width → spacing → width
4. **Error conditions**: negative/zero inputs

## File Structure

```
slicer/src/flow/
├── AGENTS.md          # This file
└── mod.rs             # Flow struct and all calculations
```

## Dependencies

- `std::f64::consts::PI` for π
- `thiserror` for error types
- Project's `Coord`/`CoordF` types and scaling functions

## Related Modules

- `gcode/path.rs` - Uses Flow for extrusion path calculations (NEEDS UPDATE)
- `pipeline/mod.rs` - Uses Flow for E-value calculations (NEEDS UPDATE)  
- `perimeter/` - Uses Flow for perimeter spacing
- `infill/` - Uses Flow for infill line spacing
- `support/` - Uses Flow for support extrusion