# Slicer API Quick Reference

## Core Types

### Pipeline

```rust
use slicer::pipeline::{PrintPipeline, PipelineConfig};

// Create pipeline with config
let config = PipelineConfig::default();
let pipeline = PrintPipeline::new(config);

// Process mesh to G-code
let gcode = pipeline.process(&mesh)?;
```

### PipelineConfig Builder Methods

```rust
PipelineConfig::default()
    // Layer settings
    .layer_height(0.2)              // mm
    .first_layer_height(0.3)        // mm
    
    // Perimeters
    .perimeters(3)                  // count
    .perimeter_mode(PerimeterMode::Classic)
    .arachne()                      // shorthand for Arachne mode
    .classic_perimeters()           // shorthand for Classic mode
    
    // Infill
    .infill_density(0.2)            // 0.0 - 1.0
    
    // Hardware
    .nozzle_diameter(0.4)           // mm
    .extruder_temperature(210)      // °C
    .bed_temperature(60)            // °C
    
    // Arc fitting (G2/G3)
    .arc_fitting(true)              // enable
    .arc_fitting_tolerance(0.05)    // mm
```

### Mesh Loading

```rust
use slicer::mesh::TriangleMesh;

// From STL file
let mesh = TriangleMesh::from_stl("model.stl")?;

// Primitives
let cube = TriangleMesh::cube(10.0);  // 10mm cube
```

### G-code Output

```rust
use slicer::gcode::GCode;

// Save to file
gcode.write_to_file("output.gcode")?;

// Get as string
let content = gcode.content();

// Statistics
let stats = gcode.stats;
println!("Layers: {}", stats.layer_count);
println!("Filament: {:.2}m", stats.filament_used_meters());
println!("Time: {}", stats.print_time_formatted());
```

---

## Configuration Enums

### InfillPattern

```rust
use slicer::config::InfillPattern;

InfillPattern::Rectilinear  // Parallel lines
InfillPattern::Grid         // Cross-hatch
InfillPattern::Concentric   // Inward loops
InfillPattern::Honeycomb    // Hexagonal
InfillPattern::Gyroid       // 3D minimal surface
InfillPattern::Lightning    // Sparse tree structure
InfillPattern::None         // No infill
```

### PerimeterMode

```rust
use slicer::config::PerimeterMode;

PerimeterMode::Classic   // Fixed-width perimeters
PerimeterMode::Arachne   // Variable-width perimeters
```

### SeamPosition

```rust
use slicer::config::SeamPosition;

SeamPosition::Random    // Random placement
SeamPosition::Nearest   // Nearest to previous
SeamPosition::Aligned   // Aligned across layers
SeamPosition::Rear      // Back of model
SeamPosition::Hidden    // Sharp corners
```

### GCodeFlavor

```rust
use slicer::config::GCodeFlavor;

GCodeFlavor::Marlin
GCodeFlavor::RepRap
GCodeFlavor::Klipper
```

---

## Infill Generation (Direct)

```rust
use slicer::infill::{InfillGenerator, InfillConfig, InfillPattern};
use slicer::geometry::ExPolygon;

let config = InfillConfig {
    pattern: InfillPattern::Gyroid,
    density: 0.2,
    extrusion_width: 0.45,
    angle: 45.0,
    ..Default::default()
};

let generator = InfillGenerator::new(config);
let result = generator.generate(&expolygons, layer_index);

// Convenience functions
use slicer::infill::*;
let result = generate_infill(&area, layer);
let result = generate_gyroid_infill(&area, density, layer);
let result = generate_honeycomb_infill(&area, density, layer);
let result = generate_lightning_infill(&area, layer, density);
```

---

## Perimeter Generation (Direct)

### Classic Mode

```rust
use slicer::perimeter::{PerimeterGenerator, PerimeterConfig};

let config = PerimeterConfig {
    perimeter_count: 3,
    perimeter_extrusion_width: 0.45,
    external_perimeters_first: false,
    ..Default::default()
};

let generator = PerimeterGenerator::new(config);
let result = generator.generate(&expolygons);

// result.perimeters - Vec of perimeter loops
// result.infill_area - Remaining area for infill
```

### Arachne Mode

```rust
use slicer::perimeter::arachne::{ArachneGenerator, ArachneConfig, BeadingStrategy};

let config = ArachneConfig {
    bead_width_outer: 0.42,
    bead_width_inner: 0.45,
    wall_count: 3,
    beading_strategy: BeadingStrategy::Distributed,
    ..Default::default()
};

let generator = ArachneGenerator::new(config);
let result = generator.generate(&expolygons);

// result.toolpaths - Vec<Vec<ExtrusionLine>>
// result.inner_contour - Area for infill
// result.thin_fills - Thin wall fills
```

---

## Arc Fitting

```rust
use slicer::gcode::{ArcFitter, ArcFittingConfig, PathSegment};

let config = ArcFittingConfig::default()
    .tolerance(0.05)      // 50 microns
    .min_radius(0.5)      // mm
    .max_radius(1000.0)   // mm
    .enabled(true);

let fitter = ArcFitter::new(config);
let segments = fitter.process_polyline(&polyline);

for segment in segments {
    match segment {
        PathSegment::Line(points) => { /* G1 moves */ }
        PathSegment::Arc(arc) => {
            // arc.direction - Clockwise/CounterClockwise
            // arc.i, arc.j - Center offsets
            // arc.to_gcode(e, f) - Generate G2/G3 string
        }
    }
}
```

---

## G-code Comparison

```rust
use slicer::gcode::{compare_gcode_files, ComparisonConfig, GCodeComparator};

// Quick comparison
let result = compare_gcode_files("expected.gcode", "actual.gcode", &config)?;

// Detailed comparison
let comparator = GCodeComparator::new(config);
let parsed_a = ParsedGCode::from_file("a.gcode")?;
let parsed_b = ParsedGCode::from_file("b.gcode")?;
let result = comparator.compare(&parsed_a, &parsed_b);

// Check results
if result.is_equivalent() {
    println!("Files match semantically");
}
println!("Total extrusion diff: {:.3}mm", result.extrusion_difference);
```

### ComparisonConfig

```rust
ComparisonConfig::default()
    .position_tolerance(0.01)   // mm for X/Y/Z
    .extrusion_tolerance(0.001) // mm for E
    .speed_tolerance(1.0)       // mm/s for F
```

---

## Pressure Equalizer

```rust
use slicer::{PressureEqualizer, PressureEqualizerConfig, PressureEqualizerStats};

// Configure the equalizer
let config = PressureEqualizerConfig::new(1.75)  // filament diameter in mm
    .with_max_slope(2.0)                          // mm³/s²
    .with_max_positive_slope(1.8)                 // rate increase limit
    .with_max_negative_slope(1.8)                 // rate decrease limit
    .with_relative_e(false);                      // absolute E mode

// Create equalizer and process G-code
let mut equalizer = PressureEqualizer::new(config);
let output = equalizer.process(gcode_input, true);  // true = flush buffer

// Get statistics
let stats = equalizer.stats();
println!("Min rate: {:.2} mm³/min", stats.volumetric_rate_min);
println!("Max rate: {:.2} mm³/min", stats.volumetric_rate_max);
println!("Avg rate: {:.2} mm³/min", stats.volumetric_rate_avg);
```

### How It Works

The pressure equalizer smooths rapid changes in volumetric extrusion rate:

1. Parses G-code into a circular buffer for lookahead
2. Calculates volumetric rate: `filament_area × feedrate × (E_length / XYZ_length)`
3. Limits rate changes with backward/forward passes
4. Splits long segments for smoother acceleration profiles
5. Outputs adjusted G-code with modified feedrates

### Special Cases

- **Bridge infill**: Unlimited slope (needs consistent flow)
- **Gap fill**: Unlimited slope (thin features need precise flow)
- **Default slope**: ~1.8 mm³/s² (20→60 mm/s over 2 seconds)

---

## Ironing

```rust
use slicer::{IroningConfig, IroningGenerator, IroningType, IroningResult};

// Configure ironing
let config = IroningConfig::new()
    .top_surfaces()                    // or .topmost_only() or .all_solid()
    .with_flow(10.0)                   // 10% flow
    .with_spacing(0.1)                 // 0.1mm line spacing
    .with_speed(20.0)                  // 20mm/s
    .with_inset(0.2)                   // 0.2mm from edge
    .with_nozzle_diameter(0.4)
    .with_layer_height(0.2);

// Generate ironing paths
let generator = IroningGenerator::new(config);
let result = generator.generate(&top_surfaces, None, layer_index, has_layer_above);

// Use paths
for path in result.paths {
    let length = path.length();
    let volume = path.volume();
    let time = path.time();
}
```

### IroningType Options

```rust
use slicer::IroningType;

IroningType::NoIroning    // Disabled
IroningType::TopSurfaces  // Iron all top surfaces
IroningType::TopmostOnly  // Only topmost layer (no layer above)
IroningType::AllSolid     // Iron all solid surfaces
```

### How It Works

1. Identifies top surfaces (based on IroningType)
2. Shrinks surfaces by inset to avoid edge artifacts
3. Generates tight rectilinear fill pattern (0.1mm spacing)
4. Extrudes with very low flow (10-15%)
5. Nozzle heat melts and smooths the surface

---

## Geometry Types

### Point (Scaled Integer)

```rust
use slicer::geometry::Point;
use slicer::{scale, unscale};

let p = Point::new(scale(10.0), scale(20.0));  // 10mm, 20mm
let x_mm = unscale(p.x);  // Back to mm

let p = Point::new_scale(10.0, 20.0);  // Convenience
```

### PointF (Floating Point)

```rust
use slicer::geometry::PointF;

let p = PointF::new(10.0, 20.0);  // Already in mm
```

### Polygon

```rust
use slicer::geometry::Polygon;

let polygon = Polygon::from_points(vec![p1, p2, p3, p4]);
let area = polygon.area();
let perimeter = polygon.perimeter();
let is_ccw = polygon.is_counter_clockwise();
```

### ExPolygon (Polygon with Holes)

```rust
use slicer::geometry::ExPolygon;

let expoly = ExPolygon::new(outer_contour);
expoly.add_hole(hole_polygon);

let contains = expoly.contains_point(&point);
let bbox = expoly.bounding_box();
```

### Polyline

```rust
use slicer::geometry::Polyline;

let polyline = Polyline::from_points(points);
let length = polyline.length();
let first = polyline.first_point();
let last = polyline.last_point();
```

---

## Clipper Operations

```rust
use slicer::clipper::*;

// Boolean operations
let union = union_expolygons(&a, &b);
let diff = difference_expolygons(&a, &b);
let inter = intersection_expolygons(&a, &b);

// Offset
let grown = offset_expolygons(&expoly, 1.0, OffsetJoinType::Miter);
let shrunk = shrink(&expoly, 0.5, OffsetJoinType::Round);
let grown = grow(&expoly, 0.5, OffsetJoinType::Square);
```

---

## Error Handling

```rust
use slicer::{Result, Error};

fn process() -> Result<GCode> {
    let mesh = TriangleMesh::from_stl("model.stl")?;
    let pipeline = PrintPipeline::with_defaults();
    pipeline.process(&mesh)
}

match process() {
    Ok(gcode) => { /* success */ }
    Err(Error::Io(e)) => { /* IO error */ }
    Err(Error::Mesh(msg)) => { /* Mesh error */ }
    Err(Error::Slice(msg)) => { /* Slicing error */ }
    Err(e) => { /* Other error */ }
}
```

---

## Seam Placer

```rust
use slicer::{SeamPlacer, SeamPlacerConfig, SeamPositionMode, LayerOutline, PerimeterOutline};

// Configure seam placement
let config = SeamPlacerConfig::aligned();  // or ::nearest(), ::random(), ::rear(), ::hidden()

// Create seam placer
let mut placer = SeamPlacer::new(config);

// Initialize with layer outlines
let outlines: Vec<LayerOutline> = layers.iter().map(|(z, polygons)| {
    LayerOutline {
        z_height: *z,
        perimeters: polygons.iter().map(|p| PerimeterOutline {
            polygon: p.clone(),
            flow_width: 0.4,
            is_external: true,
        }).collect(),
        layer_regions: vec![],  // For embedded distance calculation
    }
}).collect();

placer.init(&outlines);

// Get seam point for a perimeter
let seam_index = placer.get_seam_point(layer_idx, &polygon, Some(current_pos));

// Get 3D seam position
if let Some(pos) = placer.get_seam_position_3d(layer_idx, perimeter_idx) {
    println!("Seam at ({:.2}, {:.2}, {:.2})", pos.x, pos.y, pos.z);
}

// Simple initialization shorthand
placer.init_simple(&[(0.2, vec![polygon1]), (0.4, vec![polygon2])], 0.4);
```

### SeamPositionMode Options

```rust
use slicer::SeamPositionMode;

SeamPositionMode::Aligned  // Align seams vertically across layers (default)
SeamPositionMode::Nearest  // Place seam nearest to current position
SeamPositionMode::Random   // Deterministic random to scatter seam artifacts
SeamPositionMode::Rear     // Place seam at rear (highest Y)
SeamPositionMode::Hidden   // Actively seek concave corners
```

### SeamPlacerConfig Options

```rust
let config = SeamPlacerConfig::default()
    // Mode presets
    .seam_position(SeamPositionMode::Aligned)
    .angle_importance(0.6)           // Weight of angle in scoring
    .min_arm_length(0.5)             // Angle calc arm length (mm)
    .sharp_angle_threshold(0.96)     // ~55 degrees in radians
    .overhang_angle_threshold(0.79)  // ~45 degrees
    .seam_align_score_tolerance(0.3) // Alignment quality tolerance
    .seam_align_minimum_string_seams(6)  // Min layers for alignment
    .avoid_overhangs(true)           // Avoid overhang regions
    .prefer_hidden_points(true);     // Prefer embedded points
```

### Convenience Functions

```rust
use slicer::{place_seam, create_seam_placer};

// Quick seam placement without full initialization
let seam_idx = place_seam(&polygon, SeamPositionMode::Hidden, Some(current_pos));

// Create and initialize placer in one step
let placer = create_seam_placer(&layers, flow_width, SeamPositionMode::Aligned);
```

### How It Works

1. **Candidate Generation**: Creates candidates at each polygon vertex
2. **Angle Calculation**: Computes local CCW angle (concave = negative)
3. **Overhang Detection**: Compares against previous layer
4. **Embedded Distance**: Uses EdgeGrid for distance inside merged regions
5. **Seam Selection**: Scores using enforced/blocked, overhang, angle, visibility
6. **Layer Alignment**: Groups nearby seams across layers and adjusts positions

### Scoring Priority

1. **Point Type**: Enforced > Neutral > Blocked (user preferences)
2. **Overhang**: Prefer points with solid support beneath
3. **Embedded**: Prefer hidden points inside print (negative distance)
4. **Angle**: Concave corners have lower penalty than convex
5. **Distance**: For nearest mode, closer to current position is better

---

## Wipe Tower (Multi-Material)

```rust
use slicer::{WipeTower, WipeTowerConfig, FilamentParameters, ToolChangeResult};

// Configure wipe tower
let config = WipeTowerConfig {
    pos_x: 170.0,           // Tower X position (mm)
    pos_y: 125.0,           // Tower Y position (mm)
    width: 60.0,            // Tower width (mm)
    brim_width: 2.0,        // Brim width (mm)
    rotation_angle: 0.0,    // Rotation (degrees)
    travel_speed: 150.0,    // Travel speed (mm/s)
    first_layer_speed: 30.0,// First layer speed (mm/s)
    max_speed: 100.0,       // Maximum print speed (mm/s)
    ..Default::default()
};

// Create tower with initial tool and number of filaments
let mut tower = WipeTower::new(config, 0, 4);  // Start with tool 0, 4 filaments

// Set filament parameters
let params = FilamentParameters {
    material: "PLA".to_string(),
    nozzle_temperature: 210,
    retract_length: 0.8,
    retract_speed: 35.0,
    ..Default::default()
};
tower.set_extruder(0, params);

// Plan tool changes (call for each tool change in print)
tower.plan_toolchange(
    0.2,    // z height
    0.2,    // layer height
    0,      // old tool
    1,      // new tool
    50.0,   // purge volume for extruder change (mm³)
    30.0,   // purge volume for nozzle change (mm³)
    0.0,    // additional purge volume
);

// Generate tower G-code
let results: Vec<Vec<ToolChangeResult>> = tower.generate();

// Process results per layer
for layer_results in results {
    for result in layer_results {
        println!("Tool change {} -> {}", result.initial_tool, result.new_tool);
        println!("G-code:\n{}", result.gcode);
        println!("Elapsed time: {:.1}s", result.elapsed_time);
    }
}
```

### WipeTowerConfig Options

```rust
WipeTowerConfig {
    // Position & Size
    pos_x: 170.0,                    // X position (mm)
    pos_y: 125.0,                    // Y position (mm)
    width: 60.0,                     // Tower width (mm)
    depth: 0.0,                      // Calculated automatically
    height: 0.0,                     // Set by print height
    brim_width: 2.0,                 // First layer brim (mm)
    rotation_angle: 0.0,             // Rotation (degrees)
    
    // Speeds
    travel_speed: 150.0,             // Travel speed (mm/s)
    first_layer_speed: 30.0,         // First layer (mm/s)
    max_speed: 100.0,                // Max print speed (mm/s)
    
    // Multi-extruder settings
    semm: false,                     // Single extruder multi-material
    is_multi_extruder: false,        // Multiple extruders
    
    // G-code flavor
    gcode_flavor: WipeTowerGCodeFlavor::Marlin,
    
    // Bed shape
    bed_shape: BedShape::Rectangular,
    bed_width: 256.0,
    
    ..Default::default()
}
```

### FilamentParameters

```rust
FilamentParameters {
    material: "PLA".to_string(),
    category: 0,                     // Adhesiveness category
    is_soluble: false,
    is_support: false,
    nozzle_temperature: 200,
    nozzle_temperature_initial_layer: 210,
    nozzle_diameter: 0.4,
    retract_length: 0.8,
    retract_speed: 35.0,
    wipe_dist: 1.0,
    ramming_speed: vec![],           // Ramming profile
    max_e_speed: f32::MAX,
    ..Default::default()
}
```

### ToolChangeResult Fields

```rust
struct ToolChangeResult {
    print_z: f32,              // Z height
    layer_height: f32,         // Layer height
    gcode: String,             // Generated G-code
    extrusions: Vec<Extrusion>,// Path preview data
    start_pos: Vec2f,          // Start position
    end_pos: Vec2f,            // End position
    elapsed_time: f32,         // Time for this operation
    priming: bool,             // Is priming operation
    is_tool_change: bool,      // Is tool change
    wipe_path: Vec<Vec2f>,     // Wipe path for post-processing
    purge_volume: f32,         // Volume purged
    initial_tool: i32,         // Starting tool
    new_tool: i32,             // Ending tool
}
```

### Key Methods

```rust
// Tower depth limits based on height (for stability)
let min_depth = WipeTower::get_limit_depth_by_height(100.0);  // ~15mm for 100mm tower

// Auto brim width
let brim = WipeTower::get_auto_brim_by_height(100.0);

// Set layer before operations
tower.set_layer(z, layer_height, num_tool_changes, is_first, is_last);

// Check status
let finished = tower.finished();
let layer_done = tower.layer_finished();

// Get tower info
let width = tower.width();
let depth = tower.get_depth();
let brim = tower.get_brim_width();
let bbox = tower.get_bounding_box();

// Priming operations
let prime_results = tower.prime(&[0, 1, 2]);  // Prime tools 0, 1, 2
```

### How It Works

1. **Planning Phase**:
   - `plan_toolchange()` records each tool change with purge volumes
   - `plan_tower()` calculates total depth based on all purge needs
   - Extra spacing applied for structural stability

2. **Generation Phase**:
   - `generate()` produces G-code for all layers
   - Each layer: tool changes + layer finish (infill + perimeter)

3. **Tool Change Sequence**:
   - Retract old filament
   - Ramming (fast extrusion to push out old material)
   - Tool change command (T#)
   - Load new filament
   - Wiping (back-and-forth to clean and prime)

### Multi-Extruder Support

```rust
// Set filament mapping (which filament uses which extruder)
tower.set_filament_map(vec![0, 0, 1, 1]);  // Filaments 0,1 on extruder 0; 2,3 on extruder 1

// Check if TPU filament present (affects spacing)
tower.set_has_tpu_filament(true);
```

---

## Tool Ordering (Multi-Extruder)

```rust
use slicer::{
    ToolOrdering, ToolOrderingConfig, LayerTools, FlushMatrix,
    FilamentChangeStats, FilamentChangeMode, CustomGCodeItem,
    optimize_extruder_sequence, calculate_flush_volume,
};

// Configure tool ordering
let mut config = ToolOrderingConfig::new(4);  // 4 filaments

// Set flush matrix (purge volumes between filaments in mm³)
config.flush_matrix = FlushMatrix::new(4, 140.0);  // Default 140mm³
config.flush_matrix.set(0, 1, 100.0);  // Red to Blue: 100mm³
config.flush_matrix.set(1, 0, 120.0);  // Blue to Red: 120mm³

// Set filament properties
config.filament_types = vec!["PLA".into(), "PLA".into(), "PETG".into(), "TPU".into()];
config.filament_colors = vec!["#FF0000".into(), "#0000FF".into(), "#00FF00".into(), "#FFFF00".into()];
config.filament_soluble = vec![false, false, false, false];
config.filament_is_support = vec![false, false, false, false];

// Enable features
config.enable_prime_tower = true;
config.infill_first = false;

// Create tool ordering
let mut ordering = ToolOrdering::new(config);

// Initialize from layer Z heights
ordering.initialize_layers(vec![0.2, 0.4, 0.6, 0.8, 1.0]);

// Add extruder requirements per layer
ordering.add_extruder_to_layer(0.2, 0, true);  // Layer 1: extruder 0 (object)
ordering.add_extruder_to_layer(0.2, 1, true);  // Layer 1: extruder 1 (object)
ordering.add_extruder_to_layer(0.4, 1, true);  // Layer 2: extruder 1
ordering.add_extruder_to_layer(0.4, 2, true);  // Layer 2: extruder 2

// Handle "don't care" extruders and optimize order
ordering.handle_dontcare_extruders(Some(0));  // Start with extruder 0
ordering.reorder_extruders_for_minimum_flush();

// Collect statistics
ordering.collect_extruder_statistics(false);
ordering.fill_wipe_tower_partitions(0.0);
ordering.calculate_filament_change_stats();

// Get statistics
let stats = ordering.get_filament_change_stats(FilamentChangeMode::SingleExt);
println!("Filament changes: {}", stats.filament_change_count);
println!("Flush weight: {}g", stats.filament_flush_weight);

// Access layer tools
for layer in ordering.iter() {
    println!("Z={:.2}: extruders={:?}, wipe_partitions={}",
        layer.print_z, layer.extruders, layer.wipe_tower_partitions);
}

// Get first/last extruders
println!("First extruder: {:?}", ordering.first_extruder());
println!("Last extruder: {:?}", ordering.last_extruder());
```

### ToolOrderingConfig Options

```rust
ToolOrderingConfig {
    // Basic settings
    num_filaments: 4,
    nozzle_diameters: vec![0.4; 4],
    
    // Filament properties
    filament_densities: vec![1.24; 4],       // g/cm³
    filament_soluble: vec![false; 4],
    filament_is_support: vec![false; 4],
    filament_types: vec!["PLA".into(); 4],
    filament_colors: vec!["#FFFFFF".into(); 4],
    
    // Flush matrix and multipliers
    flush_matrix: FlushMatrix::new(4, 140.0),
    flush_multipliers: vec![1.0; 4],
    
    // Features
    enable_prime_tower: true,
    infill_first: false,
    timelapse_smooth: false,
    
    // Manual sequences (1-based extruder IDs)
    first_layer_print_sequence: vec![1, 2, 3, 4],
    other_layers_print_sequences: vec![
        ((5, 10), vec![2, 1, 3, 4]),  // Layers 5-10: custom order
    ],
    
    // Layer constraints
    max_layer_height: 0.3,
    
    ..Default::default()
}
```

### FlushMatrix Operations

```rust
// Create flush matrix
let mut matrix = FlushMatrix::new(4, 140.0);  // 4 filaments, 140mm³ default

// Set specific transitions
matrix.set(0, 1, 100.0);  // From filament 0 to 1: 100mm³
matrix.set(1, 2, 80.0);   // From filament 1 to 2: 80mm³

// Get flush volume
let flush = matrix.get(0, 1);  // 100.0

// Create from flat array (row-major)
let values = vec![0.0, 100.0, 150.0, 100.0, 0.0, 120.0, 150.0, 120.0, 0.0];
let matrix = FlushMatrix::from_flat(3, &values);

// Apply multiplier
matrix.apply_multiplier(1.5);  // All values * 1.5

// Calculate total for sequence
let sequence = vec![0, 1, 2, 0];
let total = matrix.total_flush_for_sequence(&sequence);
```

### LayerTools Fields

```rust
struct LayerTools {
    print_z: f64,                    // Z height
    has_object: bool,                // Has object extrusions
    has_support: bool,               // Has support extrusions
    extruders: Vec<u32>,             // Ordered extruder IDs (0-based)
    extruder_override: u32,          // Layer-wide override (0 = none)
    has_skirt: bool,                 // Print skirt at this layer
    has_wipe_tower: bool,            // Wipe tower active
    wipe_tower_partitions: usize,    // Number of partitions needed
    wipe_tower_layer_height: f64,    // Tower layer height
    custom_gcode: Option<CustomGCodeItem>,  // Custom G-code event
}
```

### Custom G-code Events

```rust
// Color change
let color_change = CustomGCodeItem::color_change(10.0, 1, "#FF0000");

// Pause
let pause = CustomGCodeItem::pause(20.0);

// Custom G-code
let custom = CustomGCodeItem::custom(30.0, "M400 ; wait for moves to finish");

// Assign to layers
let custom_gcodes = vec![color_change, pause, custom];
ordering.assign_custom_gcodes(&custom_gcodes);
```

### Optimization Helpers

```rust
// Optimize single layer with greedy algorithm
let extruders = vec![0, 1, 2, 3];
let optimized = optimize_extruder_sequence(&extruders, Some(0), &matrix);

// Calculate flush for sequence
let flush = calculate_flush_volume(&optimized, &matrix);

// Exhaustive search for small sets (≤8 extruders)
use slicer::find_optimal_ordering_exhaustive;
let best = find_optimal_ordering_exhaustive(&extruders, Some(0), &matrix);

// Generate all permutations (warning: O(n!))
use slicer::generate_all_orderings;
let all = generate_all_orderings(&vec![0, 1, 2]);  // 6 orderings
```

### Integration with Wipe Tower

```rust
// Use tool ordering to configure wipe tower
let ordering = ToolOrdering::new(config);
// ... initialize and optimize ...

// Get layer info for wipe tower
for layer in ordering.iter() {
    if layer.has_wipe_tower {
        // Plan tool changes for this layer
        for i in 0..layer.extruders.len().saturating_sub(1) {
            let old_tool = layer.extruders[i];
            let new_tool = layer.extruders[i + 1];
            let flush = ordering.config().get_flush_volume(
                old_tool as usize, 
                new_tool as usize
            );
            
            wipe_tower.plan_toolchange(
                layer.print_z,
                layer.wipe_tower_layer_height,
                old_tool as i32,
                new_tool as i32,
                flush,
                0.0,  // nozzle change volume
                0.0,  // additional volume
            );
        }
    }
}
```

---

## Constants

```rust
use slicer::SCALING_FACTOR;  // 1_000_000.0

// Type aliases
type Coord = i64;    // Scaled integer coordinate
type CoordF = f64;   // Float coordinate (mm)
```
