# Slicer Data Files

This directory contains structured data for 3D printers and filament materials used by the slicer.

This directory lives at the root of the slicer project (`slicer/data/`), not inside `lib/`.

## Directory Structure

```
slicer/
├── data/                  # This directory
│   ├── README.md          # This file
│   ├── schemas/           # JSON Schema definitions
│   │   ├── printer.schema.json
│   │   └── filament.schema.json
│   ├── printers/          # Printer profiles
│   │   └── <brand>-<model>-<variant>.json
│   ├── filaments/         # Filament profiles
│   │   └── <brand>-<material>-<variant>.json
│   ├── reference_gcodes/  # Reference G-code files for validation
│   └── test_stls/         # Test STL files
└── lib/                   # Rust library code
```

## Schemas

All profiles are validated against JSON Schema definitions in the `schemas/` directory.

### Printer Schema (`printer.schema.json`)

Defines the structure for printer profiles including:
- Build volume dimensions
- Nozzle specifications
- Extruder configuration
- Bed and enclosure settings
- Machine limits (speed, acceleration, jerk)
- Retraction defaults
- G-code flavor and macros
- Feature flags (auto bed leveling, input shaping, etc.)

### Filament Schema (`filament.schema.json`)

Defines the structure for filament profiles including:
- Physical properties (density, diameter, shrinkage)
- Temperature settings (nozzle, bed by surface type, chamber)
- Flow and volumetric limits
- Cooling fan settings
- Retraction overrides
- Multi-material settings
- Special properties (abrasive, flexible, requires enclosure, etc.)
- Environmental considerations (VOC emissions, ventilation)

## File Naming Convention

### Printers
```
<brand>-<model>-<variant>.json
```

Examples:
- `bambu-lab-x1-carbon-0.4.json`
- `prusa-mk4-0.4.json`
- `voron-2.4-350-0.4.json`

### Filaments
```
<brand>-<material>-<variant>.json
```

Examples:
- `generic-pla.json`
- `bambu-lab-abs.json`
- `polymaker-polyterra-pla-matte.json`
- `esun-pla-plus.json`

Use lowercase with hyphens. The `id` field inside the JSON must match the filename (without `.json`).

## Creating New Profiles

### 1. Copy an existing profile
Start from a similar existing profile to ensure you have the correct structure.

### 2. Update the `id` field
Must be unique and match the filename.

### 3. Fill in required fields

**For printers:**
- `id`, `brand`, `model`
- `build_volume` (x, y, z dimensions)
- `nozzle` (at minimum, `diameter`)

**For filaments:**
- `id`, `brand`, `material`
- `temperatures.nozzle` (at minimum, `default`)

### 4. Validate against schema
```bash
# Using ajv-cli (npm install -g ajv-cli)
ajv validate -s schemas/printer.schema.json -d printers/your-printer.json
ajv validate -s schemas/filament.schema.json -d filaments/your-filament.json

# Or using jsonschema (pip install jsonschema)
jsonschema -i printers/your-printer.json schemas/printer.schema.json
```

### 5. Test with the slicer
```bash
# From the slicer/ directory
cargo run --bin slicer-cli -- slice \
  --printer data/printers/your-printer.json \
  --filament data/filaments/your-filament.json \
  --input model.stl \
  --output output.gcode
```

## Source Attribution

When profiles are derived from other slicers (BambuStudio, PrusaSlicer, etc.), include the `source` field:

```json
{
  "source": {
    "origin": "BambuStudio",
    "version": "1.9",
    "url": "https://github.com/bambulab/BambuStudio"
  }
}
```

## Design Principles

### 1. Flat Structure (No Inheritance)
Unlike BambuStudio profiles which use `inherits` for profile composition, our profiles are fully resolved/flattened. This makes them:
- Easier to parse and validate
- Self-contained (no need to resolve inheritance chains)
- More portable across systems

### 2. Semantic Units
All values use standard units:
- Lengths: millimeters (mm)
- Temperatures: degrees Celsius (°C)
- Speeds: mm/s
- Accelerations: mm/s²
- Volumetric flow: mm³/s
- Percentages: 0-100 integer or 0.0-1.0 ratio (context-dependent, documented per field)

### 3. Optional Fields
Most fields are optional. The slicer uses sensible defaults when values aren't specified. Only specify values that differ from standard defaults.

### 4. Compatibility Arrays
Both printers and filaments have compatibility arrays (`compatible_filaments` / `compatible_printers`) that reference profile IDs. These are advisory and help with UI filtering but don't prevent manual combinations.

## Material Types Reference

Common material types supported:
- **PLA variants**: PLA, PLA+, PLA-CF, PLA-GF
- **PETG variants**: PETG, PETG-CF, PETG-GF, PCTG
- **ABS variants**: ABS, ABS-CF, ABS-GF
- **ASA variants**: ASA, ASA-CF, ASA-GF
- **Nylon/PA**: PA, PA6, PA-CF, PA-GF
- **Polycarbonate**: PC, PC-CF, PC-ABS
- **Flexible**: TPU, TPE
- **Support**: PVA, BVOH, HIPS
- **Engineering**: PP, PP-CF, PP-GF, PPA, PPA-CF, PPS, PPS-CF, POM, PEEK, PEI
- **Other**: EVA, PE, PE-CF, PHA

## Contributing

1. Fork the repository
2. Add your profile(s) following the conventions above
3. Validate against schemas
4. Test with sample prints
5. Submit a pull request with:
   - The new profile file(s)
   - A note about the source (manufacturer specs, empirical testing, etc.)
   - Any testing you performed

## Converting from BambuStudio Profiles

BambuStudio profiles use inheritance and array-wrapped values. To convert:

1. Resolve the full inheritance chain (follow `inherits` fields)
2. Merge all values (child overrides parent)
3. Unwrap array values (e.g., `["0.4"]` → `0.4`)
4. Reorganize into our schema structure
5. Add any missing required fields

A conversion script may be added in the future to automate this process.