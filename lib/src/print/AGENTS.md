# Print Module

## Purpose

The `print` module defines the core `Print` and `PrintObject` types that represent a print job and its constituent objects. These types serve as the central data structures that coordinate the slicing pipeline, holding configuration, geometry, and results.

## What This Module Does

1. **Print Job Management**: Represent a complete print job with multiple objects
2. **Object Representation**: Individual `PrintObject` instances with per-object settings
3. **State Tracking**: Track processing state through the slicing pipeline
4. **Result Aggregation**: Collect slicing results across all objects

## libslic3r Reference

| Rust Type | C++ File(s) | Description |
|-----------|-------------|-------------|
| `Print` | `Print.hpp/cpp` | Main print job container |
| `PrintObject` | `Print.hpp`, `PrintObject.cpp` | Per-object data |
| | `PrintBase.hpp/cpp` | Base class for print types |
| | `PrintRegion.cpp` | Region handling |

### Key C++ Classes

- **`Print`**: Top-level print job, holds objects and config
- **`PrintObject`**: Single object to be printed
- **`PrintRegion`**: Region with specific settings (e.g., different material)
- **`PrintBase`**: Abstract base for print types (FDM, SLA)
- **`PrintState`**: Tracks processing milestones

## Data Model

```
Print (Job)
├── config: PrintConfig           # Global settings
├── objects: Vec<PrintObject>     # Objects to print
├── regions: Vec<PrintRegion>     # Distinct print regions
└── state: PrintState             # Processing state

PrintObject
├── model_object: reference       # Source geometry
├── config: PrintObjectConfig     # Object-specific settings
├── layers: Vec<Layer>            # Sliced layers
├── support_layers: Vec<Layer>    # Support layers
└── state: ObjectState            # Object processing state
```

## Key Types

### Print

```rust
pub struct Print {
    /// Global print configuration
    pub config: PrintConfig,
    
    /// Objects to be printed
    pub objects: Vec<PrintObject>,
    
    /// Distinct print regions (for multi-material)
    pub regions: Vec<PrintRegion>,
    
    /// Processing state
    state: PrintState,
    
    /// Estimated print time
    pub estimated_time: Option<f64>,
    
    /// Estimated filament usage
    pub filament_used: Option<f64>,
}
```

### PrintObject

```rust
pub struct PrintObject {
    /// Object identifier
    pub id: usize,
    
    /// Source mesh
    pub mesh: TriangleMesh,
    
    /// Object-specific configuration
    pub config: PrintObjectConfig,
    
    /// Transformation matrix
    pub transform: Transform3D,
    
    /// Sliced layers (populated after slicing)
    pub layers: Vec<Layer>,
    
    /// Support layers (if support enabled)
    pub support_layers: Vec<SupportLayer>,
    
    /// Object bounding box
    pub bounding_box: BoundingBox3F,
}
```

### PrintRegion

```rust
pub struct PrintRegion {
    /// Region identifier
    pub id: usize,
    
    /// Region-specific configuration
    pub config: PrintRegionConfig,
    
    /// Which objects/parts use this region
    pub object_ids: Vec<usize>,
}
```

### PrintState

```rust
pub enum PrintState {
    /// Initial state, nothing processed
    New,
    
    /// Objects have been sliced
    Sliced,
    
    /// Perimeters generated
    Perimeters,
    
    /// Infill generated
    Infill,
    
    /// Support generated
    Support,
    
    /// G-code generated
    GCodeGenerated,
    
    /// Export complete
    Exported,
}
```

## Print Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        Print::new()                              │
│                    Create print with config                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    print.add_object(mesh)                        │
│                  Add objects to print job                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      print.validate()                            │
│              Validate configuration and objects                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      print.process()                             │
│         Run full pipeline: slice → perimeters → infill           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    print.export_gcode()                          │
│                   Generate G-code output                         │
└─────────────────────────────────────────────────────────────────┘
```

## API

### Print Methods

```rust
impl Print {
    /// Create a new print job with configuration
    pub fn new(config: PrintConfig) -> Self;
    
    /// Add an object to the print
    pub fn add_object(&mut self, mesh: TriangleMesh) -> &mut PrintObject;
    
    /// Add object with specific configuration
    pub fn add_object_with_config(
        &mut self, 
        mesh: TriangleMesh, 
        config: PrintObjectConfig
    ) -> &mut PrintObject;
    
    /// Validate the print job
    pub fn validate(&self) -> Result<(), PrintError>;
    
    /// Process all objects (slice, perimeters, infill, support)
    pub fn process(&mut self) -> Result<(), PrintError>;
    
    /// Generate G-code
    pub fn export_gcode(&self) -> Result<GCode, PrintError>;
    
    /// Get processing state
    pub fn state(&self) -> PrintState;
    
    /// Get all objects
    pub fn objects(&self) -> &[PrintObject];
    
    /// Get mutable objects
    pub fn objects_mut(&mut self) -> &mut [PrintObject];
    
    /// Calculate print statistics
    pub fn calculate_stats(&mut self);
}
```

### PrintObject Methods

```rust
impl PrintObject {
    /// Get object bounding box
    pub fn bounding_box(&self) -> &BoundingBox3F;
    
    /// Get sliced layers
    pub fn layers(&self) -> &[Layer];
    
    /// Get layer at index
    pub fn layer(&self, index: usize) -> Option<&Layer>;
    
    /// Get layer count
    pub fn layer_count(&self) -> usize;
    
    /// Apply transformation
    pub fn set_transform(&mut self, transform: Transform3D);
    
    /// Set position on bed
    pub fn set_position(&mut self, x: f64, y: f64);
    
    /// Set rotation around Z axis
    pub fn set_rotation(&mut self, angle: f64);
    
    /// Set scale factor
    pub fn set_scale(&mut self, scale: f64);
}
```

## File Structure

```
slicer/src/print/
├── AGENTS.md    # This file
└── mod.rs       # Print and PrintObject types
```

## Multi-Object Prints

A print can contain multiple objects:

```rust
let mut print = Print::new(config);

// Add multiple objects
let obj1 = print.add_object(mesh1);
obj1.set_position(50.0, 50.0);

let obj2 = print.add_object(mesh2);
obj2.set_position(150.0, 50.0);

// Process all together
print.process()?;
```

Objects can have different configurations:

```rust
let obj1 = print.add_object_with_config(mesh1, PrintObjectConfig {
    layer_height: 0.1,  // Fine detail
    ..Default::default()
});

let obj2 = print.add_object_with_config(mesh2, PrintObjectConfig {
    layer_height: 0.3,  // Fast draft
    ..Default::default()
});
```

## Region System

Regions allow different settings for different parts:

- **Different materials**: Multi-color or multi-material prints
- **Modifier meshes**: Override settings in specific areas
- **Paint-on settings**: User-defined regions

```rust
// Create region with specific settings
let region = PrintRegion {
    id: 1,
    config: PrintRegionConfig {
        infill_density: 1.0,  // 100% infill in this region
        ..Default::default()
    },
    object_ids: vec![0],  // Applies to first object
};
```

## Processing Steps

The `process()` method runs these steps:

```rust
impl Print {
    pub fn process(&mut self) -> Result<(), PrintError> {
        // 1. Slice all objects
        for object in &mut self.objects {
            object.slice()?;
        }
        self.state = PrintState::Sliced;
        
        // 2. Generate perimeters
        for object in &mut self.objects {
            object.generate_perimeters()?;
        }
        self.state = PrintState::Perimeters;
        
        // 3. Prepare infill (classify surfaces)
        for object in &mut self.objects {
            object.prepare_infill()?;
        }
        
        // 4. Generate infill
        for object in &mut self.objects {
            object.generate_infill()?;
        }
        self.state = PrintState::Infill;
        
        // 5. Generate support (if enabled)
        for object in &mut self.objects {
            if object.config.support_enabled {
                object.generate_support()?;
            }
        }
        self.state = PrintState::Support;
        
        Ok(())
    }
}
```

## Object Arrangement

Objects are arranged on the bed with:
- Position (X, Y translation)
- Rotation (Z-axis rotation)
- Scale (uniform or per-axis)

The Print coordinates object placement to avoid collisions.

## Dependencies

- `crate::config::{PrintConfig, PrintObjectConfig, PrintRegionConfig}` - Configs
- `crate::mesh::TriangleMesh` - Source geometry
- `crate::slice::Layer` - Sliced layer data
- `crate::support::SupportLayer` - Support data
- `crate::geometry::{BoundingBox3F, Transform3D}` - Spatial types
- `crate::gcode::GCode` - Output type

## Related Modules

- `config/` - Configuration types used by Print
- `mesh/` - Source mesh data
- `slice/` - Layer slicing
- `perimeter/` - Perimeter generation
- `infill/` - Infill generation
- `support/` - Support generation
- `pipeline/` - Alternative processing orchestration
- `gcode/` - G-code output

## Testing Strategy

1. **Unit tests**: Object creation, state transitions
2. **Multi-object tests**: Multiple objects process correctly
3. **Config tests**: Per-object configs override globals
4. **Validation tests**: Invalid configs rejected
5. **Integration tests**: Full pipeline with real meshes

## Print vs Pipeline

The codebase has two ways to orchestrate slicing:

| Feature | `Print` | `PrintPipeline` |
|---------|---------|-----------------|
| Multi-object | Yes | Single object |
| Per-object config | Yes | Single config |
| State tracking | Detailed | Simple |
| Complexity | Higher | Lower |
| Use case | Full prints | Testing/simple cases |

For full print jobs, use `Print`. For simple single-object slicing, `PrintPipeline` may be simpler.

## Future Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| Basic Print type | ✅ Done | Single object support |
| PrintObject | ✅ Done | Object representation |
| Multi-object | 🔄 Partial | Multiple objects |
| Object transforms | 🔄 Partial | Position/rotate/scale |
| Print regions | 📋 Planned | Multi-material support |
| Sequential printing | 📋 Planned | Print objects one at a time |
| Object arrangement | 📋 Planned | Auto-arrange on bed |
| Collision detection | 📋 Planned | Prevent object overlap |