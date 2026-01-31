# Pipeline Module

## Purpose

The `pipeline` module orchestrates the complete slicing process, coordinating all other modules to transform a 3D mesh into G-code. It is the main entry point for slicing operations and ensures all steps are executed in the correct order with proper data flow between stages.

## What This Module Does

1. **Pipeline Configuration (`PipelineConfig`)**: Aggregates all configuration from print, object, and region configs into a single coherent structure.

2. **Slicing Orchestration (`PrintPipeline`)**: Executes the full pipeline: mesh → slices → perimeters → infill → support → paths → G-code.

3. **Layer Processing**: Iterates through layers, generating perimeters, infill, and support for each.

4. **G-code Generation**: Converts toolpaths to G-code using proper extrusion calculations.

5. **Travel Path Optimization**: Uses `AvoidCrossingPerimeters` to route travel moves around perimeter walls, reducing visible marks on printed surfaces.

6. **Statistics Collection**: Tracks print time estimates, filament usage, layer counts.

## libslic3r Reference

This module corresponds to several high-level C++ classes:

| Rust | C++ File(s) | Description |
|------|-------------|-------------|
| `PipelineConfig` | `PrintConfig.hpp`, `PrintObjectConfig.hpp` | Configuration aggregation |
| `PrintPipeline` | `Print.cpp`, `PrintObject.cpp` | Main slicing orchestration |
| `PrintPipeline::slice()` | `Print::process()` | Full pipeline execution |
| `PrintPipeline::generate_gcode()` | `GCode::do_export()` | G-code generation |
| `calculate_e_for_distance()` | `Flow::mm3_per_mm()` + E calculation | Extrusion math |

## Critical: Extrusion Calculation

The `calculate_e_for_distance()` method uses the **exact formula from libslic3r/Flow.cpp**:

```
cross_section = height × (width - height × (1 - π/4))
volume = cross_section × distance
E = volume / (π × (filament_diameter/2)²)
```

This is NOT `width × height × 0.9` or any other approximation. The formula models the actual physical shape of extruded plastic (rectangle with semicircular ends).

For bridges, use `calculate_e_for_bridge()` which uses circular cross-section.

## Pipeline Stages

```
Input: TriangleMesh + PipelineConfig
         │
         ▼
    ┌─────────────┐
    │   Slicing   │  → Layer[] with slice polygons
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Perimeters  │  → PerimeterResult per layer
    └─────────────┘    (Classic or Arachne mode)
         │
         ▼
    ┌─────────────┐
    │   Infill    │  → InfillResult per layer
    └─────────────┘    (various patterns)
         │
         ▼
    ┌─────────────┐
    │   Support   │  → SupportLayer[] 
    └─────────────┘    (if enabled)
         │
         ▼
    ┌─────────────┐
    │   Bridges   │  → Bridge detection & flow
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │    Paths    │  → LayerPaths[] ordered toolpaths
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   G-code    │  → GCode with commands
    └─────────────┘    (with travel optimization)
         │
         ▼
Output: GCode (with stats)
```

## Travel Path Optimization

The pipeline integrates `AvoidCrossingPerimeters` for travel path planning:

1. **Per-Layer Initialization**: At the start of each layer, perimeter polygons are extracted from the layer paths and used to initialize the travel planner's boundaries.

2. **Travel Routing**: When a travel move is needed between paths, `emit_travel()` routes the move through the travel planner, which may modify the path to avoid crossing perimeters.

3. **Configuration**: Controlled via `PrintConfig`:
   - `avoid_crossing_perimeters: bool` - Enable/disable (default: true)
   - `avoid_crossing_max_detour: f64` - Max detour as % of direct path (default: 200%)

4. **Fallback**: If the detour would exceed the max allowed, the direct path is used instead.

## Key Methods

| Method | Purpose |
|--------|---------|
| `PrintPipeline::new()` | Create pipeline with config |
| `PrintPipeline::process()` | Execute full pipeline (takes `&mut self`) |
| `PrintPipeline::generate_gcode()` | Generate G-code from processed layers |
| `write_layer()` | Write G-code for a single layer with travel optimization |
| `emit_travel()` | Emit travel move, routing through travel planner if enabled |
| `calculate_e_for_distance()` | Calculate E value for normal extrusion |
| `calculate_e_for_bridge()` | Calculate E value for bridge extrusion |
| `extrude_path_linear()` | Write linear path to G-code |
| `extrude_path_with_arcs()` | Write path with arc fitting |

## File Structure

```
slicer/src/pipeline/
├── AGENTS.md    # This file
└── mod.rs       # All pipeline implementation
```

## Configuration Hierarchy

```
PipelineConfig
├── print: PrintConfig          # Machine & general settings
│   ├── bed_size_x/y
│   ├── nozzle_diameter
│   ├── filament_diameter      # Critical for E calculations!
│   ├── extrusion_multiplier
│   └── speeds, temperatures...
│
├── object: PrintObjectConfig   # Per-object settings
│   ├── layer_height
│   ├── perimeters
│   ├── infill_density
│   └── support settings...
│
└── region: PrintRegionConfig   # Per-region settings
    ├── extrusion widths
    ├── speeds by role
    └── infill patterns...
```

## Error Handling

The pipeline returns `Result<GCode, Error>` where errors can be:
- `Error::Mesh` - Invalid input mesh
- `Error::Slicing` - Slicing algorithm failure
- `Error::Config` - Invalid configuration
- `Error::Cancelled` - User cancellation

## Thread Safety

The pipeline is designed to be single-threaded per print job, but internal operations (infill generation, perimeter generation) use Rayon for parallelism where beneficial.

## Dependencies

- `crate::mesh` - Input mesh
- `crate::slice` - Layer slicing
- `crate::perimeter` - Perimeter generation (Classic & Arachne)
- `crate::infill` - Infill pattern generation
- `crate::support` - Support structure generation
- `crate::bridge` - Bridge detection
- `crate::gcode` - G-code writing
- `crate::flow` - Extrusion calculations
- `crate::adhesion` - Brim/skirt generation
- `crate::travel` - Travel path optimization (AvoidCrossingPerimeters)
- `crate::geometry` - Geometric operations
- `crate::clipper` - Boolean operations

## Related Modules

- `../flow/` - Flow calculations used for extrusion math
- `../gcode/` - G-code generation primitives
- `../config/` - Configuration structures
- `../print/` - Print and PrintObject types
- `../travel/` - Travel path planning (AvoidCrossingPerimeters)
- `../edge_grid/` - Spatial acceleration for travel planning

## Testing Strategy

1. **Unit Tests**: Test individual helper methods
2. **Integration Tests**: `tests/benchy_integration.rs` runs full pipeline
3. **Parity Tests**: Compare output against BambuStudio reference G-code

## Performance Considerations

- Layer processing can be parallelized with Rayon
- G-code generation is sequential (stateful writer)
- Memory: One layer's paths kept in memory at a time during G-code generation