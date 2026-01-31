# Config Module

## Purpose

The `config` module defines all configuration types for the slicing process. These structures mirror BambuStudio/PrusaSlicer's configuration system and control every aspect of how a model is sliced and printed - from layer heights to temperatures to speeds.

## What This Module Does

1. **Print Configuration** (`print_config.rs`): Machine and global print settings
2. **Region Configuration** (`region_config.rs`): Per-region/per-object settings
3. **Configuration Validation**: Ensure settings are valid and consistent
4. **Default Values**: Provide sensible defaults matching BambuStudio

## libslic3r Reference

| Rust File | C++ File(s) | Description |
|-----------|-------------|-------------|
| `mod.rs` | - | Module exports |
| `print_config.rs` | `PrintConfig.hpp/cpp` | Main configuration definitions |
| `region_config.rs` | `PrintConfig.hpp` | Region-specific settings |
| | `Config.hpp/cpp` | Base configuration system |
| | `Preset.hpp/cpp` | Preset management |

### Key C++ Classes

- **`PrintConfig`**: Global print settings
- **`PrintObjectConfig`**: Per-object settings
- **`PrintRegionConfig`**: Per-region settings (materials, etc.)
- **`DynamicPrintConfig`**: Runtime-modifiable config
- **`ConfigOption`**: Base class for config values
- **`ConfigOptionDef`**: Option definitions with metadata

## Configuration Hierarchy

```
PrintConfig (Machine/Global)
├── Bed size, shape
├── Nozzle diameter
├── Filament settings
├── Temperatures
├── G-code flavor
└── Machine limits

PrintObjectConfig (Per-Object)
├── Layer heights
├── Perimeter count
├── Infill settings
├── Support settings
└── Seam position

PrintRegionConfig (Per-Region)
├── Extrusion widths
├── Speeds by feature type
├── Flow ratios
├── Infill patterns
└── Bridge settings
```

## Key Types

### PrintConfig

Global/machine-level settings:

```rust
pub struct PrintConfig {
    // === Bed ===
    pub bed_size_x: f64,              // Bed width (mm)
    pub bed_size_y: f64,              // Bed depth (mm)
    pub bed_shape: BedShape,          // Rectangular, circular, custom
    
    // === Nozzle ===
    pub nozzle_diameter: f64,         // Nozzle size (mm), typically 0.4
    
    // === Filament ===
    pub filament_diameter: f64,       // Filament diameter (mm), 1.75 or 2.85
    pub extrusion_multiplier: f64,    // Flow rate adjustment
    
    // === Temperatures ===
    pub first_layer_bed_temp: u32,    // First layer bed temperature
    pub bed_temperature: u32,         // Regular bed temperature
    pub first_layer_temp: u32,        // First layer extruder temp
    pub temperature: u32,             // Regular extruder temperature
    
    // === Speeds ===
    pub travel_speed: f64,            // Travel move speed (mm/s)
    pub first_layer_speed: f64,       // First layer speed (mm/s)
    pub print_speed: f64,             // Default print speed (mm/s)
    
    // === Retraction ===
    pub retract_length: f64,          // Retraction distance (mm)
    pub retract_speed: f64,           // Retraction speed (mm/s)
    pub retract_lift: f64,            // Z lift on retraction (mm)
    
    // === Layers ===
    pub layer_height: f64,            // Default layer height (mm)
    pub first_layer_height: f64,      // First layer height (mm)
    
    // === G-code ===
    pub gcode_flavor: GCodeFlavor,    // Marlin, Klipper, etc.
    pub use_relative_e: bool,         // Relative extrusion mode
    pub use_firmware_retraction: bool, // G10/G11 retraction
}
```

### PrintObjectConfig

Per-object settings:

```rust
pub struct PrintObjectConfig {
    // === Layers ===
    pub layer_height: f64,
    pub first_layer_height: f64,
    
    // === Perimeters ===
    pub perimeters: u32,              // Number of shells
    pub perimeter_mode: PerimeterMode, // Classic or Arachne
    
    // === Infill ===
    pub infill_density: f64,          // 0.0 - 1.0
    pub infill_pattern: InfillPattern,
    pub top_solid_layers: u32,
    pub bottom_solid_layers: u32,
    
    // === Support ===
    pub support_enabled: bool,
    pub support_type: SupportType,
    pub support_threshold_angle: f64,
    
    // === Quality ===
    pub seam_position: SeamPosition,
    pub external_perimeters_first: bool,
}
```

### PrintRegionConfig

Per-region settings (detailed extrusion parameters):

```rust
pub struct PrintRegionConfig {
    // === Extrusion Widths ===
    pub external_perimeter_extrusion_width: f64,
    pub perimeter_extrusion_width: f64,
    pub infill_extrusion_width: f64,
    pub solid_infill_extrusion_width: f64,
    pub top_infill_extrusion_width: f64,
    
    // === Speeds ===
    pub external_perimeter_speed: f64,
    pub perimeter_speed: f64,
    pub infill_speed: f64,
    pub solid_infill_speed: f64,
    pub top_solid_infill_speed: f64,
    pub bridge_speed: f64,
    pub travel_speed: f64,
    
    // === Flow Ratios ===
    pub bridge_flow_ratio: f64,
    pub infill_overlap: f64,
    
    // === Patterns ===
    pub fill_pattern: InfillPattern,
    pub top_fill_pattern: InfillPattern,
    pub bottom_fill_pattern: InfillPattern,
    pub fill_angle: f64,
}
```

## Enums

### GCodeFlavor

```rust
pub enum GCodeFlavor {
    Marlin,
    Klipper,
    RepRap,
    Smoothieware,
    Mach3,
}
```

### PerimeterMode

```rust
pub enum PerimeterMode {
    Classic,    // Fixed-width perimeters
    Arachne,    // Variable-width perimeters
}
```

### InfillPattern

```rust
pub enum InfillPattern {
    Rectilinear,
    Grid,
    Honeycomb,
    Gyroid,
    Concentric,
    Lightning,
    // ... more patterns
}
```

### SeamPosition

```rust
pub enum SeamPosition {
    Random,
    Nearest,
    Aligned,
    Rear,
    Custom,
}
```

## File Structure

```
slicer/src/config/
├── AGENTS.md           # This file
├── mod.rs              # Module exports
├── print_config.rs     # PrintConfig type
└── region_config.rs    # PrintRegionConfig type
```

## Default Values

Defaults are chosen to match BambuStudio/PrusaSlicer profiles:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `layer_height` | 0.2mm | Common default |
| `first_layer_height` | 0.3mm | Better adhesion |
| `nozzle_diameter` | 0.4mm | Standard nozzle |
| `filament_diameter` | 1.75mm | Most common |
| `perimeters` | 3 | Good strength |
| `infill_density` | 0.15 (15%) | Light infill |
| `print_speed` | 60mm/s | Moderate speed |
| `travel_speed` | 150mm/s | Fast travel |

## Validation

Configurations are validated to catch common errors:

```rust
impl PrintConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.layer_height <= 0.0 {
            return Err(ConfigError::InvalidValue("layer_height must be positive"));
        }
        if self.layer_height > self.nozzle_diameter {
            return Err(ConfigError::InvalidValue("layer_height > nozzle_diameter"));
        }
        if self.first_layer_height <= 0.0 {
            return Err(ConfigError::InvalidValue("first_layer_height must be positive"));
        }
        if self.filament_diameter <= 0.0 {
            return Err(ConfigError::InvalidValue("filament_diameter must be positive"));
        }
        // ... more validations
        Ok(())
    }
}
```

## Configuration Sources

In a full implementation, configs come from:

1. **Built-in defaults**: `Default::default()`
2. **Preset files**: JSON/INI profiles
3. **Command-line arguments**: CLI overrides
4. **Per-object settings**: Model-specific overrides
5. **Modifier meshes**: Region-specific overrides

## libslic3r Config System

The C++ config system is more complex:

```cpp
// ConfigOption hierarchy
ConfigOption
├── ConfigOptionFloat
├── ConfigOptionInt
├── ConfigOptionBool
├── ConfigOptionString
├── ConfigOptionFloatOrPercent  // "50%" or "0.4"
├── ConfigOptionEnum
└── ConfigOptionPoints          // Multiple points

// Definition includes metadata
ConfigOptionDef {
    type, default_value, min, max,
    label, tooltip, category,
    aliases, shortcut
}
```

Our Rust implementation simplifies this by using plain structs with typed fields.

## Auto Width Calculation

When extrusion width is set to 0 (auto), it's calculated from nozzle diameter:

```rust
fn auto_extrusion_width(nozzle_diameter: f64, role: FlowRole) -> f64 {
    match role {
        FlowRole::ExternalPerimeter => nozzle_diameter * 1.125,
        FlowRole::Perimeter => nozzle_diameter * 1.125,
        FlowRole::Infill => nozzle_diameter * 1.125,
        FlowRole::SolidInfill => nozzle_diameter * 1.125,
        FlowRole::TopSolidInfill => nozzle_diameter,
        FlowRole::SupportMaterial => nozzle_diameter,
        // ... etc
    }
}
```

## Dependencies

- `crate::infill::InfillPattern` - Infill pattern enum
- `crate::gcode::SeamPosition` - Seam position enum
- Standard library for defaults

## Related Modules

- `pipeline/` - Uses configs to orchestrate slicing
- `flow/` - Uses widths/heights for flow calculations
- `perimeter/` - Uses perimeter settings
- `infill/` - Uses infill settings
- `support/` - Uses support settings
- `gcode/` - Uses G-code flavor and speeds

## Testing Strategy

1. **Default tests**: Defaults are valid and sensible
2. **Validation tests**: Invalid configs are rejected
3. **Serialization tests**: Round-trip to/from JSON (future)
4. **Preset tests**: Load standard presets correctly
5. **Override tests**: Per-object settings override globals

## Future Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| Basic PrintConfig | ✅ Done | Core machine settings |
| PrintObjectConfig | ✅ Done | Per-object settings |
| PrintRegionConfig | ✅ Done | Per-region settings |
| Validation | ✅ Done | Basic validation |
| JSON serialization | 📋 Planned | Load/save configs |
| Preset system | 📋 Planned | Named preset profiles |
| Config inheritance | 📋 Planned | Presets inherit from parents |
| Percent values | 📋 Planned | "50%" of layer height |
| Config GUI bindings | 📋 Planned | For UI integration |