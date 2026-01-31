# Profiles Module

## Purpose

The profiles module provides structured data types and utilities for loading, validating, and managing 3D printer and filament profiles. These profiles replace hardcoded configuration values with real-world machine specifications and material properties.

## Architecture

```
profiles/
├── mod.rs          # Main module with all profile types and registry
└── AGENTS.md       # This documentation
```

## Key Types

### Printer Profiles (`PrinterProfile`)

Represents a 3D printer with all its specifications:

- **Build volume**: X, Y, Z dimensions and origin position
- **Nozzle**: Diameter, type (brass/hardened/etc.), max temperature
- **Extruder**: Type (direct drive/bowden), filament diameter, count
- **Bed**: Heated, max temperature, surface types
- **Enclosure**: Enclosed, heated chamber, air filtration
- **Limits**: Max speed, acceleration, jerk per axis
- **Retraction**: Default length, speed, z-hop, wipe settings
- **G-code**: Flavor, start/end macros, layer change scripts
- **Features**: Auto bed leveling, pressure advance, input shaping, etc.

### Filament Profiles (`FilamentProfile`)

Represents a filament material with printing parameters:

- **Material type**: PLA, PETG, ABS, etc.
- **Physical properties**: Density, diameter, shrinkage, hygroscopic
- **Temperatures**: Nozzle (min/max/default), bed (per surface type), chamber
- **Flow**: Ratio multiplier, max volumetric speed
- **Cooling**: Fan speeds, layer time thresholds, overhang settings
- **Retraction overrides**: Filament-specific retraction settings
- **Multi-material**: Prime volume, purge settings, flush temperature
- **Special properties**: Matte, silk, carbon fiber, requires enclosure, etc.
- **Environmental**: Ventilation, VOC emissions, food safety

### Profile Registry (`ProfileRegistry`)

Manages collections of loaded profiles:

- Load all profiles from a directory
- Lookup by ID
- Find compatible printer/filament combinations
- Iterate over all loaded profiles

## Data File Format

Profiles are stored as JSON files in `data/printers/` and `data/filaments/`.

### Naming Convention

```
printers/<brand>-<model>-<variant>.json
filaments/<brand>-<material>-<variant>.json
```

Examples:
- `printers/bambu-lab-x1-carbon-0.4.json`
- `filaments/generic-pla.json`
- `filaments/bambu-lab-pla-matte.json`

### Validation

All profiles are validated against JSON Schema definitions in `data/schemas/`:
- `printer.schema.json`
- `filament.schema.json`

## Usage Examples

### Load a Single Profile

```rust
use slicer::profiles::{PrinterProfile, FilamentProfile};

let printer = PrinterProfile::from_file("data/printers/bambu-lab-x1-carbon-0.4.json")?;
let filament = FilamentProfile::from_file("data/filaments/generic-pla.json")?;

println!("Build volume: {}x{}x{}mm", 
    printer.build_volume.x,
    printer.build_volume.y,
    printer.build_volume.z
);
println!("Nozzle temp: {}°C", filament.nozzle_temperature());
```

### Load All Profiles from Directory

```rust
use slicer::profiles::ProfileRegistry;

let registry = ProfileRegistry::load_from_directory("data")?;

println!("Loaded {} printers, {} filaments", 
    registry.printer_count(),
    registry.filament_count()
);

// Look up specific profiles
if let Some(printer) = registry.get_printer("bambu-lab-x1-carbon-0.4") {
    println!("Found: {} {}", printer.brand, printer.model);
}

// Find compatible combinations
let compatible = registry.compatible_filaments("bambu-lab-x1-carbon-0.4");
for filament in compatible {
    println!("Compatible: {} {}", filament.brand, filament.material);
}
```

### Helper Methods

```rust
// Printer helpers
printer.has_heated_bed();           // -> bool
printer.is_enclosed();              // -> bool
printer.max_print_speed();          // -> Option<f64>
printer.retraction_length();        // -> Option<f64>
printer.filament_diameter();        // -> f64

// Filament helpers
filament.nozzle_temperature();              // -> f64
filament.first_layer_nozzle_temperature();  // -> f64
filament.bed_temperature();                 // -> Option<f64>
filament.flow_ratio();                      // -> f64
filament.max_volumetric_speed();            // -> Option<f64>
filament.requires_enclosure();              // -> bool
filament.requires_hardened_nozzle();        // -> bool
filament.is_support_material();             // -> bool
filament.density();                         // -> Option<f64>
filament.diameter();                        // -> f64
```

## Design Decisions

### Flat Structure (No Inheritance)

Unlike BambuStudio's profile system which uses `inherits` for composition, our profiles are fully resolved/flattened. This makes them:

1. **Self-contained**: No need to resolve inheritance chains
2. **Portable**: Work without access to parent profiles
3. **Simpler to parse**: Direct deserialization, no post-processing
4. **Easier to validate**: All required fields present

### Optional Fields with Sensible Defaults

Most fields are optional. Serde's `#[serde(default)]` provides defaults, and helper methods return `Option<T>` for truly optional values.

### Compatibility Arrays

Both printers and filaments have `compatible_printers`/`compatible_filaments` arrays that reference profile IDs. These are:
- Advisory (don't prevent manual combinations)
- Bidirectional (both sides can declare compatibility)
- Used for UI filtering and suggestions

### Source Attribution

The `source` field tracks where a profile came from (BambuStudio, PrusaSlicer, custom) for proper attribution and debugging.

## Relationship to BambuStudio

This module was designed by analyzing BambuStudio's profile structure:

| BambuStudio | Our Format |
|-------------|------------|
| `machine/*.json` | `printers/*.json` |
| `filament/*.json` | `filaments/*.json` |
| `inherits` field | (Resolved/flattened) |
| Array-wrapped values (`["0.4"]`) | Direct values (`0.4`) |
| Separate nozzle variant files | Single file with variant in ID |

## Converting BambuStudio Profiles

To convert from BambuStudio format:

1. Resolve inheritance chain (follow `inherits`)
2. Merge values (child overrides parent)
3. Unwrap array values
4. Reorganize into our schema structure
5. Add required fields if missing

A conversion script may be added in the future.

## Testing

Run profile tests:

```bash
cargo test profiles::
```

Tests include:
- JSON parsing validation
- Loading actual data files
- Profile registry operations
- Helper method behavior
- Validation error handling