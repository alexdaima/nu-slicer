//! Slice configuration combining printer, filament, and print settings.
//!
//! This module provides a unified configuration system that:
//! - References printer and filament profiles by ID
//! - Allows selecting specific nozzle diameter
//! - Provides override settings for quality, speed, etc.
//! - Can be serialized to/from JSON for config files

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::{FilamentProfile, PrinterProfile, ProfileError, ProfileRegistry, ProfileResult};
use crate::config::{
    GCodeFlavor, InfillPattern, PerimeterMode, PrintConfig, PrintObjectConfig, SeamPosition,
    SupportType,
};
use crate::pipeline::PipelineConfig;
use crate::slice::SlicingParams;
use crate::support::{SupportConfig, SupportPattern};

/// Complete slice configuration referencing profiles and overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceConfig {
    /// Schema reference for validation
    #[serde(rename = "$schema", skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,

    /// Printer profile ID (e.g., "bambu-lab-h2d")
    pub printer_id: String,

    /// Filament profile ID (e.g., "bambu-pla-basic")
    pub filament_id: String,

    /// Nozzle diameter to use (defaults to printer's default_nozzle_diameter)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nozzle_diameter: Option<f64>,

    /// Quality settings
    #[serde(default)]
    pub quality: QualitySettings,

    /// Speed settings
    #[serde(default)]
    pub speed: SpeedSettings,

    /// Perimeter/shell settings
    #[serde(default)]
    pub perimeters: PerimeterSettings,

    /// Infill settings
    #[serde(default)]
    pub infill: InfillSettings,

    /// Support settings
    #[serde(default)]
    pub support: SupportSettings,

    /// Adhesion settings (brim, skirt, raft)
    #[serde(default)]
    pub adhesion: AdhesionSettings,

    /// G-code settings
    #[serde(default)]
    pub gcode: GCodeSettings,

    /// Advanced settings
    #[serde(default)]
    pub advanced: AdvancedSettings,

    /// Custom overrides as key-value pairs for extensibility
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Quality-related settings (layer heights, resolution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Layer height in mm (if not set, uses a sensible default based on nozzle diameter)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_height: Option<f64>,

    /// First layer height in mm (if not set, defaults to layer_height)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_layer_height: Option<f64>,

    /// Resolution for G-code output in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<f64>,

    /// Slice closing radius in mm (for healing small gaps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slice_closing_radius: Option<f64>,

    /// XY size compensation in mm (positive = larger, negative = smaller)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xy_size_compensation: Option<f64>,

    /// Elephant foot compensation in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elephant_foot_compensation: Option<f64>,
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            layer_height: None,
            first_layer_height: None,
            resolution: None,
            slice_closing_radius: None,
            xy_size_compensation: None,
            elephant_foot_compensation: None,
        }
    }
}

/// Speed-related settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedSettings {
    /// Default print speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub print_speed: Option<f64>,

    /// Travel move speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub travel_speed: Option<f64>,

    /// First layer speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_layer_speed: Option<f64>,

    /// Perimeter speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perimeter_speed: Option<f64>,

    /// External perimeter speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_perimeter_speed: Option<f64>,

    /// Infill speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infill_speed: Option<f64>,

    /// Solid infill speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solid_infill_speed: Option<f64>,

    /// Top solid infill speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_solid_infill_speed: Option<f64>,

    /// Bridge speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bridge_speed: Option<f64>,

    /// Gap fill speed in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap_fill_speed: Option<f64>,
}

impl Default for SpeedSettings {
    fn default() -> Self {
        Self {
            print_speed: None,
            travel_speed: None,
            first_layer_speed: None,
            perimeter_speed: None,
            external_perimeter_speed: None,
            infill_speed: None,
            solid_infill_speed: None,
            top_solid_infill_speed: None,
            bridge_speed: None,
            gap_fill_speed: None,
        }
    }
}

/// Perimeter/shell settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerimeterSettings {
    /// Number of perimeter loops
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,

    /// Number of top solid layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_solid_layers: Option<u32>,

    /// Number of bottom solid layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bottom_solid_layers: Option<u32>,

    /// Perimeter generation mode ("classic" or "arachne")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,

    /// Enable thin walls detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thin_walls: Option<bool>,

    /// Enable gap fill
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap_fill: Option<bool>,

    /// Enable overhang detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overhangs: Option<bool>,

    /// Seam position ("random", "aligned", "rear", "nearest", "hidden")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seam_position: Option<String>,

    /// Fuzzy skin mode ("none", "outside", "all")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuzzy_skin: Option<String>,

    /// Fuzzy skin thickness in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuzzy_skin_thickness: Option<f64>,

    /// Fuzzy skin point distance in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuzzy_skin_point_distance: Option<f64>,
}

impl Default for PerimeterSettings {
    fn default() -> Self {
        Self {
            count: None,
            top_solid_layers: None,
            bottom_solid_layers: None,
            mode: None,
            thin_walls: None,
            gap_fill: None,
            overhangs: None,
            seam_position: None,
            fuzzy_skin: None,
            fuzzy_skin_thickness: None,
            fuzzy_skin_point_distance: None,
        }
    }
}

/// Infill settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfillSettings {
    /// Infill density as percentage (0-100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<f64>,

    /// Infill pattern ("rectilinear", "grid", "honeycomb", "gyroid", "concentric", "lightning", etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,

    /// Infill angle in degrees
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,

    /// Solid infill threshold area (mm²)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solid_threshold_area: Option<f64>,
}

impl Default for InfillSettings {
    fn default() -> Self {
        Self {
            density: None,
            pattern: None,
            angle: None,
            solid_threshold_area: None,
        }
    }
}

/// Support structure settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportSettings {
    /// Enable support structures
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,

    /// Support type ("normal", "tree", "hybrid")
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub support_type: Option<String>,

    /// Overhang threshold angle in degrees
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold_angle: Option<f64>,

    /// Support density as fraction (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<f64>,

    /// Support pattern ("grid", "lines", "honeycomb", "gyroid", "lightning")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,

    /// Only generate support on build plate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buildplate_only: Option<bool>,

    /// Support Z distance in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_distance: Option<f64>,

    /// Support XY distance in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xy_distance: Option<f64>,

    /// Support interface layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interface_layers: Option<u32>,
}

impl Default for SupportSettings {
    fn default() -> Self {
        Self {
            enabled: None,
            support_type: None,
            threshold_angle: None,
            density: None,
            pattern: None,
            buildplate_only: None,
            z_distance: None,
            xy_distance: None,
            interface_layers: None,
        }
    }
}

/// Adhesion settings (brim, skirt, raft).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdhesionSettings {
    /// Adhesion type ("none", "skirt", "brim", "raft")
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub adhesion_type: Option<String>,

    /// Number of skirt loops
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skirt_loops: Option<u32>,

    /// Skirt distance from object in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skirt_distance: Option<f64>,

    /// Skirt minimum length in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skirt_min_length: Option<f64>,

    /// Brim width in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brim_width: Option<f64>,
}

impl Default for AdhesionSettings {
    fn default() -> Self {
        Self {
            adhesion_type: None,
            skirt_loops: None,
            skirt_distance: None,
            skirt_min_length: None,
            brim_width: None,
        }
    }
}

/// G-code generation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCodeSettings {
    /// Use relative E coordinates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_relative_e: Option<bool>,

    /// Enable arc fitting (G2/G3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arc_fitting: Option<bool>,

    /// Arc fitting tolerance in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arc_fitting_tolerance: Option<f64>,

    /// Arc fitting minimum radius in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arc_fitting_min_radius: Option<f64>,

    /// Arc fitting maximum radius in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arc_fitting_max_radius: Option<f64>,

    /// Enable spiral/vase mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spiral_vase: Option<bool>,
}

impl Default for GCodeSettings {
    fn default() -> Self {
        Self {
            use_relative_e: None,
            arc_fitting: None,
            arc_fitting_tolerance: None,
            arc_fitting_min_radius: None,
            arc_fitting_max_radius: None,
            spiral_vase: None,
        }
    }
}

/// Advanced/experimental settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSettings {
    /// Enable avoid crossing perimeters for travel moves
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avoid_crossing_perimeters: Option<bool>,

    /// Maximum detour percentage for avoid crossing perimeters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avoid_crossing_max_detour: Option<f64>,

    /// Enable bridge detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bridge_detection: Option<bool>,

    /// Bridge flow multiplier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bridge_flow_multiplier: Option<f64>,

    /// Bridge speed multiplier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bridge_speed_multiplier: Option<f64>,

    /// Extrusion multiplier (flow rate adjustment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extrusion_multiplier: Option<f64>,

    /// Retraction length override in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retract_length: Option<f64>,

    /// Retraction speed override in mm/s
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retract_speed: Option<f64>,

    /// Z-hop height in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retract_lift: Option<f64>,

    /// Minimum travel distance before retraction in mm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retract_before_travel: Option<f64>,
}

impl Default for AdvancedSettings {
    fn default() -> Self {
        Self {
            avoid_crossing_perimeters: None,
            avoid_crossing_max_detour: None,
            bridge_detection: None,
            bridge_flow_multiplier: None,
            bridge_speed_multiplier: None,
            extrusion_multiplier: None,
            retract_length: None,
            retract_speed: None,
            retract_lift: None,
            retract_before_travel: None,
        }
    }
}

impl SliceConfig {
    /// Create a new slice config with just printer and filament IDs.
    pub fn new(printer_id: impl Into<String>, filament_id: impl Into<String>) -> Self {
        Self {
            schema: Some("../schemas/slice_config.schema.json".to_string()),
            printer_id: printer_id.into(),
            filament_id: filament_id.into(),
            nozzle_diameter: None,
            quality: QualitySettings::default(),
            speed: SpeedSettings::default(),
            perimeters: PerimeterSettings::default(),
            infill: InfillSettings::default(),
            support: SupportSettings::default(),
            adhesion: AdhesionSettings::default(),
            gcode: GCodeSettings::default(),
            advanced: AdvancedSettings::default(),
            custom: HashMap::new(),
        }
    }

    /// Load a slice config from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> ProfileResult<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Parse a slice config from JSON string.
    pub fn from_json(json: &str) -> ProfileResult<Self> {
        serde_json::from_str(json).map_err(|e| ProfileError::Json(e))
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> ProfileResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| ProfileError::Json(e))
    }

    /// Save to a JSON file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> ProfileResult<()> {
        let content = self.to_json()?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Build a PipelineConfig from this slice config and the profile registry.
    pub fn build_pipeline_config(
        &self,
        registry: &ProfileRegistry,
    ) -> ProfileResult<PipelineConfig> {
        // Look up printer profile
        let printer = registry.get_printer(&self.printer_id).ok_or_else(|| {
            ProfileError::NotFound(format!("Printer '{}' not found", self.printer_id))
        })?;

        // Look up filament profile
        let filament = registry.get_filament(&self.filament_id).ok_or_else(|| {
            ProfileError::NotFound(format!("Filament '{}' not found", self.filament_id))
        })?;

        self.build_pipeline_config_with_profiles(printer, filament)
    }

    /// Build a PipelineConfig from this slice config and explicit profiles.
    pub fn build_pipeline_config_with_profiles(
        &self,
        printer: &PrinterProfile,
        filament: &FilamentProfile,
    ) -> ProfileResult<PipelineConfig> {
        // Determine nozzle diameter from config or printer's default
        let nozzle_diameter = self
            .nozzle_diameter
            .unwrap_or(printer.default_nozzle_diameter);

        // Get nozzle-specific config if available
        let nozzle_config = printer.get_nozzle_config(nozzle_diameter);

        // Get layer limits from nozzle config or printer profile
        let layer_limits = printer.layer_limits.as_ref();

        // Get retraction from nozzle config or printer profile
        let retraction = nozzle_config
            .and_then(|nc| nc.retraction.as_ref())
            .or(printer.retraction.as_ref());

        // Compute default layer height based on nozzle (typically 50% of nozzle diameter)
        let max_layer_height = layer_limits
            .and_then(|l| l.max_layer_height)
            .unwrap_or(nozzle_diameter * 0.7);
        let default_layer_height = (nozzle_diameter * 0.5).min(max_layer_height);

        // Build PrintConfig
        let mut print_config = PrintConfig::default();

        // Bed configuration from printer
        print_config.bed_size_x = printer.build_volume.x;
        print_config.bed_size_y = printer.build_volume.y;
        print_config.print_origin_x = if printer.build_volume.origin == "center" {
            -printer.build_volume.x / 2.0
        } else {
            0.0
        };
        print_config.print_origin_y = if printer.build_volume.origin == "center" {
            -printer.build_volume.y / 2.0
        } else {
            0.0
        };

        // Nozzle and filament
        print_config.nozzle_diameter = nozzle_diameter;
        print_config.filament_diameter = filament.diameter();

        // Layer heights
        print_config.layer_height = self.quality.layer_height.unwrap_or(default_layer_height);
        print_config.first_layer_height = self
            .quality
            .first_layer_height
            .unwrap_or(print_config.layer_height);

        // Temperatures from filament
        print_config.extruder_temperature = filament.nozzle_temperature() as u32;
        print_config.first_layer_extruder_temperature =
            filament.first_layer_nozzle_temperature() as u32;
        print_config.bed_temperature = filament.bed_temperature().unwrap_or(60.0) as u32;
        print_config.first_layer_bed_temperature =
            filament.bed_temperature().unwrap_or(60.0) as u32;

        // Speeds
        let max_print_speed = printer.max_print_speed().unwrap_or(200.0);
        print_config.print_speed = self.speed.print_speed.unwrap_or(60.0).min(max_print_speed);
        print_config.travel_speed = self
            .speed
            .travel_speed
            .unwrap_or_else(|| {
                printer
                    .limits
                    .as_ref()
                    .and_then(|l| l.max_speed.as_ref())
                    .map(|s| s.travel.unwrap_or(150.0))
                    .unwrap_or(150.0)
            })
            .min(max_print_speed * 2.0);
        print_config.first_layer_speed = self.speed.first_layer_speed.unwrap_or(20.0);

        // Retraction from printer profile or overrides
        print_config.retract_length = self
            .advanced
            .retract_length
            .unwrap_or_else(|| retraction.and_then(|r| r.length).unwrap_or(0.8));
        print_config.retract_speed = self
            .advanced
            .retract_speed
            .unwrap_or_else(|| retraction.and_then(|r| r.speed).unwrap_or(30.0));
        print_config.retract_lift = self
            .advanced
            .retract_lift
            .unwrap_or_else(|| retraction.and_then(|r| r.z_hop).unwrap_or(0.4));
        print_config.retract_before_travel = self
            .advanced
            .retract_before_travel
            .unwrap_or_else(|| retraction.and_then(|r| r.minimum_travel).unwrap_or(1.0));

        // Extrusion multiplier from filament flow ratio
        print_config.extrusion_multiplier = self
            .advanced
            .extrusion_multiplier
            .unwrap_or(filament.flow_ratio());

        // Adhesion
        print_config.skirt_loops = self.adhesion.skirt_loops.unwrap_or(3);
        print_config.skirt_distance = self.adhesion.skirt_distance.unwrap_or(6.0);
        print_config.skirt_min_length = self.adhesion.skirt_min_length.unwrap_or(0.0);
        print_config.brim_width = self.adhesion.brim_width.unwrap_or(0.0);

        // Support
        print_config.support_enabled = self.support.enabled.unwrap_or(false);
        print_config.support_type = match self.support.support_type.as_deref() {
            Some("tree") => SupportType::Tree,
            Some("hybrid") => SupportType::Hybrid,
            _ => SupportType::Normal,
        };
        print_config.support_threshold_angle = self.support.threshold_angle.unwrap_or(45.0);
        print_config.support_density = self.support.density.unwrap_or(0.15);

        // G-code settings
        print_config.gcode_flavor = match printer.gcode.as_ref().and_then(|g| g.flavor.as_deref()) {
            Some("klipper") => GCodeFlavor::Klipper,
            Some("reprap") => GCodeFlavor::RepRap,
            Some("smoothie") => GCodeFlavor::Smoothie,
            Some("sailfish") => GCodeFlavor::Sailfish,
            Some("mach3") => GCodeFlavor::Mach3,
            _ => GCodeFlavor::Marlin,
        };
        print_config.resolution = self.quality.resolution.unwrap_or(0.0125);
        print_config.use_relative_e = self.gcode.use_relative_e.unwrap_or(true);
        print_config.arc_fitting_enabled = self.gcode.arc_fitting.unwrap_or_else(|| {
            printer
                .features
                .as_ref()
                .map(|f| f.arc_support)
                .unwrap_or(true)
        });
        print_config.arc_fitting_tolerance = self.gcode.arc_fitting_tolerance.unwrap_or(0.05);
        print_config.arc_fitting_min_radius = self.gcode.arc_fitting_min_radius.unwrap_or(0.5);
        print_config.arc_fitting_max_radius = self.gcode.arc_fitting_max_radius.unwrap_or(1000.0);
        print_config.spiral_vase = self.gcode.spiral_vase.unwrap_or(false);

        // Travel optimization
        print_config.avoid_crossing_perimeters =
            self.advanced.avoid_crossing_perimeters.unwrap_or(true);
        print_config.avoid_crossing_max_detour =
            self.advanced.avoid_crossing_max_detour.unwrap_or(2.0);

        // Build PrintObjectConfig
        let mut object_config = PrintObjectConfig::default();

        object_config.layer_height = print_config.layer_height;
        object_config.perimeters = self.perimeters.count.unwrap_or(3);
        object_config.top_solid_layers = self.perimeters.top_solid_layers.unwrap_or(4);
        object_config.bottom_solid_layers = self.perimeters.bottom_solid_layers.unwrap_or(4);
        object_config.fill_density = self.infill.density.unwrap_or(15.0) / 100.0;
        object_config.fill_pattern = match self.infill.pattern.as_deref() {
            Some("grid") => InfillPattern::Grid,
            Some("honeycomb") => InfillPattern::Honeycomb,
            Some("gyroid") => InfillPattern::Gyroid,
            Some("concentric") => InfillPattern::Concentric,
            Some("triangles") => InfillPattern::Triangles,
            Some("cubic") => InfillPattern::Cubic,
            Some("lightning") => InfillPattern::Lightning,
            Some("adaptive") | Some("adaptive_cubic") => InfillPattern::AdaptiveCubic,
            _ => InfillPattern::Rectilinear,
        };

        // Speeds
        object_config.perimeter_speed = self
            .speed
            .perimeter_speed
            .unwrap_or(print_config.print_speed);
        object_config.external_perimeter_speed = self
            .speed
            .external_perimeter_speed
            .unwrap_or(object_config.perimeter_speed * 0.5);
        object_config.infill_speed = self.speed.infill_speed.unwrap_or(print_config.print_speed);
        object_config.solid_infill_speed = self
            .speed
            .solid_infill_speed
            .unwrap_or(object_config.infill_speed * 0.8);
        object_config.top_solid_infill_speed = self
            .speed
            .top_solid_infill_speed
            .unwrap_or(object_config.solid_infill_speed * 0.8);
        object_config.bridge_speed = self.speed.bridge_speed.unwrap_or(30.0);
        object_config.gap_fill_speed = self.speed.gap_fill_speed.unwrap_or(20.0);

        // Perimeter options
        object_config.thin_walls = self.perimeters.thin_walls.unwrap_or(true);
        object_config.gap_fill = self.perimeters.gap_fill.unwrap_or(true);
        object_config.overhangs = self.perimeters.overhangs.unwrap_or(true);

        object_config.seam_position = match self.perimeters.seam_position.as_deref() {
            Some("random") => SeamPosition::Random,
            Some("rear") => SeamPosition::Rear,
            Some("nearest") => SeamPosition::Nearest,
            Some("hidden") => SeamPosition::Hidden,
            _ => SeamPosition::Aligned,
        };

        object_config.perimeter_mode = match self.perimeters.mode.as_deref() {
            Some("arachne") => PerimeterMode::Arachne,
            _ => PerimeterMode::Classic,
        };

        // Quality adjustments
        object_config.slice_closing_radius = self.quality.slice_closing_radius.unwrap_or(0.049);
        object_config.xy_size_compensation = self.quality.xy_size_compensation.unwrap_or(0.0);
        object_config.elephant_foot_compensation =
            self.quality.elephant_foot_compensation.unwrap_or(0.0);

        // Build SlicingParams
        let slicing_params = SlicingParams {
            layer_height: print_config.layer_height,
            first_layer_height: print_config.first_layer_height,
            ..SlicingParams::default()
        };

        // Build SupportConfig
        let support_config = SupportConfig {
            enabled: print_config.support_enabled,
            support_type: match self.support.support_type.as_deref() {
                Some("tree") => crate::support::SupportType::Tree,
                Some("organic") => crate::support::SupportType::Organic,
                _ => crate::support::SupportType::Normal,
            },
            overhang_angle: print_config.support_threshold_angle,
            pattern: match self.support.pattern.as_deref() {
                Some("lines") | Some("rectilinear") => SupportPattern::Lines,
                Some("honeycomb") => SupportPattern::Honeycomb,
                Some("gyroid") => SupportPattern::Gyroid,
                Some("lightning") => SupportPattern::Lightning,
                _ => SupportPattern::Grid,
            },
            density: print_config.support_density,
            z_distance: self.support.z_distance.unwrap_or(0.2),
            xy_distance: self.support.xy_distance.unwrap_or(0.4),
            top_interface_layers: self.support.interface_layers.unwrap_or(2) as usize,
            buildplate_only: self.support.buildplate_only.unwrap_or(false),
            ..SupportConfig::default()
        };

        // Build the final PipelineConfig
        let mut pipeline_config = PipelineConfig {
            print: print_config,
            object: object_config,
            slicing: slicing_params,
            support: support_config,
            detect_bridges: self.advanced.bridge_detection.unwrap_or(true),
            ..PipelineConfig::default()
        };

        // Apply bridge settings
        if let Some(flow_mult) = self.advanced.bridge_flow_multiplier {
            pipeline_config.bridge.flow_multiplier = flow_mult;
        }
        if let Some(speed_mult) = self.advanced.bridge_speed_multiplier {
            pipeline_config.bridge.speed_multiplier = speed_mult;
        }

        Ok(pipeline_config)
    }

    /// Get the effective nozzle diameter (from config or printer default).
    pub fn effective_nozzle_diameter(&self, printer: &PrinterProfile) -> f64 {
        self.nozzle_diameter
            .unwrap_or(printer.default_nozzle_diameter)
    }

    /// Validate the config against the registry.
    pub fn validate(&self, registry: &ProfileRegistry) -> ProfileResult<()> {
        // Check printer exists
        if registry.get_printer(&self.printer_id).is_none() {
            return Err(ProfileError::NotFound(format!(
                "Printer '{}' not found in registry",
                self.printer_id
            )));
        }

        // Check filament exists
        if registry.get_filament(&self.filament_id).is_none() {
            return Err(ProfileError::NotFound(format!(
                "Filament '{}' not found in registry",
                self.filament_id
            )));
        }

        // Validate layer height constraints
        if let Some(layer_height) = self.quality.layer_height {
            if layer_height <= 0.0 {
                return Err(ProfileError::Invalid(
                    "layer_height must be positive".to_string(),
                ));
            }
            if layer_height > 1.0 {
                return Err(ProfileError::Invalid(
                    "layer_height seems too large (> 1mm)".to_string(),
                ));
            }
        }

        // Validate infill density
        if let Some(density) = self.infill.density {
            if density < 0.0 || density > 100.0 {
                return Err(ProfileError::Invalid(
                    "infill density must be between 0 and 100".to_string(),
                ));
            }
        }

        // Validate support threshold angle
        if let Some(angle) = self.support.threshold_angle {
            if angle < 0.0 || angle > 90.0 {
                return Err(ProfileError::Invalid(
                    "support threshold_angle must be between 0 and 90 degrees".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self::new("generic", "generic-pla")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_config_new() {
        let config = SliceConfig::new("bambu-lab-h2d", "bambu-pla-basic");
        assert_eq!(config.printer_id, "bambu-lab-h2d");
        assert_eq!(config.filament_id, "bambu-pla-basic");
        assert!(config.nozzle_diameter.is_none());
    }

    #[test]
    fn test_slice_config_serialization() {
        let mut config = SliceConfig::new("bambu-lab-h2d", "bambu-pla-basic");
        config.nozzle_diameter = Some(0.4);
        config.quality.layer_height = Some(0.2);
        config.infill.density = Some(15.0);

        let json = config.to_json().unwrap();
        assert!(json.contains("bambu-lab-h2d"));
        assert!(json.contains("bambu-pla-basic"));
        assert!(json.contains("0.2"));

        let parsed = SliceConfig::from_json(&json).unwrap();
        assert_eq!(parsed.printer_id, "bambu-lab-h2d");
        assert_eq!(parsed.quality.layer_height, Some(0.2));
    }

    #[test]
    fn test_slice_config_defaults() {
        let config = SliceConfig::default();
        assert!(config.quality.layer_height.is_none());
        assert!(config.infill.density.is_none());
        assert!(config.support.enabled.is_none());
    }

    #[test]
    fn test_quality_settings_default() {
        let quality = QualitySettings::default();
        assert!(quality.layer_height.is_none());
        assert!(quality.resolution.is_none());
    }

    #[test]
    fn test_speed_settings_default() {
        let speed = SpeedSettings::default();
        assert!(speed.print_speed.is_none());
        assert!(speed.travel_speed.is_none());
    }
}
