//! Profiles module for loading printer and filament definitions.
//!
//! This module provides types and utilities for loading printer and filament
//! profiles from JSON files, making it easy to configure the slicer with
//! real-world printer specifications and material properties.
//!
//! ## Example
//!
//! ```rust,ignore
//! use slicer::profiles::{PrinterProfile, FilamentProfile, SliceConfig, ProfileRegistry};
//!
//! // Load profiles from a data directory
//! let registry = ProfileRegistry::load_from_directory("data")?;
//!
//! // Create a slice config referencing profiles
//! let slice_config = SliceConfig::new("bambu-lab-h2d", "bambu-pla-basic");
//!
//! // Build pipeline config from profiles
//! let pipeline_config = slice_config.build_pipeline_config(&registry)?;
//!
//! // Or load from a JSON config file
//! let slice_config = SliceConfig::from_file("my_print_config.json")?;
//! ```

mod slice_config;

pub use slice_config::{
    AdhesionSettings, AdvancedSettings, GCodeSettings, InfillSettings, PerimeterSettings,
    QualitySettings, SliceConfig, SpeedSettings, SupportSettings,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Error type for profile operations.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid profile: {0}")]
    Invalid(String),

    #[error("Profile not found: {0}")]
    NotFound(String),
}

pub type ProfileResult<T> = Result<T, ProfileError>;

// ============================================================================
// Printer Profile Types
// ============================================================================

/// A 3D printer profile containing machine specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrinterProfile {
    /// Unique identifier (matches filename without .json)
    pub id: String,

    /// Printer manufacturer
    pub brand: String,

    /// Printer model name
    pub model: String,

    /// Specific variant (e.g., "0.4mm nozzle")
    #[serde(default)]
    pub variant: Option<String>,

    /// Human-readable description
    #[serde(default)]
    pub description: Option<String>,

    /// Source attribution
    #[serde(default)]
    pub source: Option<ProfileSource>,

    /// Printing technology (FFF/FDM)
    #[serde(default = "default_technology")]
    pub technology: String,

    /// Kinematic structure
    #[serde(default)]
    pub structure: Option<String>,

    /// Build volume dimensions
    pub build_volume: BuildVolume,

    /// Default nozzle diameter (mm)
    #[serde(default = "default_nozzle_diameter")]
    pub default_nozzle_diameter: f64,

    /// Nozzle configurations keyed by diameter string (e.g., "0.4")
    #[serde(default)]
    pub nozzle_configs: std::collections::HashMap<String, NozzleConfig>,

    /// Legacy single nozzle spec (for backward compatibility)
    #[serde(default)]
    pub nozzle: Option<NozzleSpec>,

    /// Extruder configuration
    #[serde(default)]
    pub extruder: Option<ExtruderConfig>,

    /// Print bed configuration
    #[serde(default)]
    pub bed: Option<BedConfig>,

    /// Enclosure specifications
    #[serde(default)]
    pub enclosure: Option<EnclosureConfig>,

    /// Machine movement limits
    #[serde(default)]
    pub limits: Option<MachineLimits>,

    /// Layer height constraints
    #[serde(default)]
    pub layer_limits: Option<LayerLimits>,

    /// Default retraction settings
    #[serde(default)]
    pub retraction: Option<RetractionConfig>,

    /// G-code configuration
    #[serde(default)]
    pub gcode: Option<GCodeConfig>,

    /// Feature flags
    #[serde(default)]
    pub features: Option<PrinterFeatures>,

    /// Extruder clearance for sequential printing
    #[serde(default)]
    pub clearance: Option<Clearance>,

    /// Part cooling configuration
    #[serde(default)]
    pub cooling: Option<CoolingConfig>,

    /// Compatible filament profile IDs
    #[serde(default)]
    pub compatible_filaments: Vec<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: Option<ProfileMetadata>,
}

/// Profile source attribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileSource {
    /// Original source (e.g., "BambuStudio", "PrusaSlicer")
    #[serde(default)]
    pub origin: Option<String>,

    /// Version of the source
    #[serde(default)]
    pub version: Option<String>,

    /// URL to the source
    #[serde(default)]
    pub url: Option<String>,

    /// Original filament ID
    #[serde(default)]
    pub filament_id: Option<String>,
}

/// Build volume dimensions in millimeters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildVolume {
    /// Width (X dimension)
    pub x: f64,

    /// Depth (Y dimension)
    pub y: f64,

    /// Height (Z dimension)
    pub z: f64,

    /// Origin position
    #[serde(default = "default_origin")]
    pub origin: String,
}

/// Nozzle configuration for a specific diameter.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NozzleConfig {
    /// Nozzle diameter (mm)
    pub diameter: f64,

    /// Nozzle type/material
    #[serde(default)]
    pub nozzle_type: Option<String>,

    /// Minimum layer height (mm)
    #[serde(default)]
    pub min_layer_height: Option<f64>,

    /// Maximum layer height (mm)
    #[serde(default)]
    pub max_layer_height: Option<f64>,

    /// Retraction settings for this nozzle
    #[serde(default)]
    pub retraction: Option<RetractionConfig>,

    /// Start G-code for this nozzle
    #[serde(default)]
    pub start_gcode: Option<String>,
}

/// Nozzle specifications (legacy format).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NozzleSpec {
    /// Nozzle diameter (mm)
    #[serde(default = "default_nozzle_diameter")]
    pub diameter: f64,

    /// Nozzle material type
    #[serde(rename = "type")]
    #[serde(default)]
    pub nozzle_type: Option<String>,

    /// Nozzle height from heat block
    #[serde(default)]
    pub height: Option<f64>,

    /// Maximum safe temperature (°C)
    #[serde(default)]
    pub max_temperature: Option<f64>,
}

/// Extruder configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtruderConfig {
    /// Extruder type (direct_drive, bowden)
    #[serde(rename = "type")]
    #[serde(default)]
    pub extruder_type: Option<String>,

    /// Expected filament diameter
    #[serde(default = "default_filament_diameter")]
    pub filament_diameter: f64,

    /// Number of extruders
    #[serde(default = "default_extruder_count")]
    pub count: u32,

    /// Gear ratio
    #[serde(default)]
    pub gear_ratio: Option<f64>,
}

/// Print bed configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BedConfig {
    /// Whether bed is heated
    #[serde(default = "default_true")]
    pub heated: bool,

    /// Maximum bed temperature (°C)
    #[serde(default)]
    pub max_temperature: Option<f64>,

    /// Available surface types
    #[serde(default)]
    pub surface_types: Vec<String>,

    /// Default surface type
    #[serde(default)]
    pub default_surface: Option<String>,
}

/// Enclosure configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnclosureConfig {
    /// Whether printer is enclosed
    #[serde(default)]
    pub enclosed: bool,

    /// Whether chamber can be heated
    #[serde(default)]
    pub heated_chamber: bool,

    /// Maximum chamber temperature (°C)
    #[serde(default)]
    pub max_chamber_temperature: Option<f64>,

    /// Air filtration available
    #[serde(default)]
    pub air_filtration: bool,
}

/// Machine movement limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MachineLimits {
    /// Maximum speeds (mm/s)
    #[serde(default)]
    pub max_speed: Option<SpeedLimits>,

    /// Maximum accelerations (mm/s²)
    #[serde(default)]
    pub max_acceleration: Option<AccelerationLimits>,

    /// Maximum jerk values (mm/s)
    #[serde(default)]
    pub max_jerk: Option<JerkLimits>,
}

/// Speed limits per axis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeedLimits {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub e: Option<f64>,
    pub travel: Option<f64>,
    pub print: Option<f64>,
}

/// Acceleration limits per axis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccelerationLimits {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub e: Option<f64>,
    pub extruding: Option<f64>,
    pub retracting: Option<f64>,
    pub travel: Option<f64>,
}

/// Jerk limits per axis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JerkLimits {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub e: Option<f64>,
}

/// Layer height constraints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerLimits {
    /// Minimum layer height (mm)
    #[serde(default)]
    pub min_layer_height: Option<f64>,

    /// Maximum layer height (mm)
    #[serde(default)]
    pub max_layer_height: Option<f64>,
}

/// Retraction configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetractionConfig {
    /// Retraction length (mm)
    #[serde(default)]
    pub length: Option<f64>,

    /// Retraction speed (mm/s)
    #[serde(default)]
    pub speed: Option<f64>,

    /// Deretraction speed (mm/s)
    #[serde(default)]
    pub deretraction_speed: Option<f64>,

    /// Z-hop height (mm)
    #[serde(default)]
    pub z_hop: Option<f64>,

    /// Minimum travel before retraction (mm)
    #[serde(default)]
    pub minimum_travel: Option<f64>,

    /// Enable wipe
    #[serde(default)]
    pub wipe: Option<bool>,

    /// Wipe distance (mm)
    #[serde(default)]
    pub wipe_distance: Option<f64>,

    /// Retract before wipe percentage
    #[serde(default)]
    pub retract_before_wipe: Option<f64>,
}

/// G-code configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GCodeConfig {
    /// G-code flavor/dialect
    #[serde(default)]
    pub flavor: Option<String>,

    /// Start G-code
    #[serde(default)]
    pub start_gcode: Option<String>,

    /// End G-code
    #[serde(default)]
    pub end_gcode: Option<String>,

    /// Before layer change G-code
    #[serde(default)]
    pub before_layer_change_gcode: Option<String>,

    /// After layer change G-code
    #[serde(default)]
    pub after_layer_change_gcode: Option<String>,

    /// Tool change G-code
    #[serde(default)]
    pub toolchange_gcode: Option<String>,

    /// Pause G-code
    #[serde(default)]
    pub pause_gcode: Option<String>,
}

/// Printer feature flags.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrinterFeatures {
    #[serde(default)]
    pub auto_bed_leveling: bool,
    #[serde(default)]
    pub pressure_advance: bool,
    #[serde(default)]
    pub input_shaping: bool,
    #[serde(default = "default_true")]
    pub arc_support: bool,
    #[serde(default)]
    pub firmware_retraction: bool,
    #[serde(default)]
    pub multi_material: bool,
    #[serde(default)]
    pub filament_runout_sensor: bool,
    #[serde(default)]
    pub power_loss_recovery: bool,
    #[serde(default)]
    pub camera: bool,
    #[serde(default)]
    pub lidar: bool,
}

/// Extruder clearance for sequential printing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Clearance {
    /// Maximum radius of extruder assembly (mm)
    #[serde(default)]
    pub radius: Option<f64>,

    /// Height clearance (mm)
    #[serde(default)]
    pub height: Option<f64>,
}

/// Part cooling configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoolingConfig {
    /// Number of cooling fans
    #[serde(default = "default_fan_count")]
    pub fan_count: u32,

    /// Maximum fan speed percentage
    #[serde(default = "default_100")]
    pub max_fan_speed: u32,

    /// Auxiliary fan present
    #[serde(default)]
    pub auxiliary_fan: bool,
}

/// Profile metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileMetadata {
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub modified: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

// ============================================================================
// Filament Profile Types
// ============================================================================

/// A filament profile containing material properties and print settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilamentProfile {
    /// Unique identifier
    pub id: String,

    /// Filament manufacturer
    pub brand: String,

    /// Base material type (PLA, PETG, ABS, etc.)
    pub material: String,

    /// Product name
    #[serde(default)]
    pub name: Option<String>,

    /// Specific variant
    #[serde(default)]
    pub variant: Option<String>,

    /// Human-readable description
    #[serde(default)]
    pub description: Option<String>,

    /// Color specification
    #[serde(default)]
    pub color: Option<FilamentColor>,

    /// Source attribution
    #[serde(default)]
    pub source: Option<ProfileSource>,

    /// Physical properties
    #[serde(default)]
    pub physical_properties: Option<PhysicalProperties>,

    /// Temperature settings
    pub temperatures: TemperatureSettings,

    /// Flow settings
    #[serde(default)]
    pub flow: Option<FlowSettings>,

    /// Cooling settings
    #[serde(default)]
    pub cooling: Option<FilamentCooling>,

    /// Retraction overrides
    #[serde(default)]
    pub retraction: Option<RetractionConfig>,

    /// Multi-material settings
    #[serde(default)]
    pub multi_material: Option<MultiMaterialSettings>,

    /// Support material properties
    #[serde(default)]
    pub support_material: Option<SupportMaterialProperties>,

    /// Special properties
    #[serde(default)]
    pub special_properties: Option<SpecialProperties>,

    /// Environmental considerations
    #[serde(default)]
    pub environment: Option<EnvironmentSettings>,

    /// Cost information
    #[serde(default)]
    pub cost: Option<CostInfo>,

    /// Compatible printer profile IDs
    #[serde(default)]
    pub compatible_printers: Vec<String>,

    /// Filament-specific G-code
    #[serde(default)]
    pub gcode: Option<FilamentGCode>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: Option<ProfileMetadata>,
}

/// Filament color specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilamentColor {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub hex: Option<String>,
}

/// Physical properties of the filament.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicalProperties {
    /// Filament diameter (mm)
    #[serde(default = "default_filament_diameter")]
    pub diameter: f64,

    /// Density (g/cm³)
    #[serde(default)]
    pub density: Option<f64>,

    /// Spool weight (g)
    #[serde(default)]
    pub spool_weight: Option<f64>,

    /// Glass transition temperature (°C)
    #[serde(default)]
    pub glass_transition_temperature: Option<f64>,

    /// Vitrification temperature (°C)
    #[serde(default)]
    pub vitrification_temperature: Option<f64>,

    /// Shrinkage percentage (100 = no shrinkage)
    #[serde(default = "default_100_f64")]
    pub shrinkage: f64,

    /// Material absorbs moisture
    #[serde(default)]
    pub hygroscopic: bool,

    /// Material is abrasive
    #[serde(default)]
    pub abrasive: bool,

    /// Material is flexible
    #[serde(default)]
    pub flexible: bool,

    /// Shore hardness (e.g., "95A")
    #[serde(default)]
    pub shore_hardness: Option<String>,

    /// Impact strength in Z (kJ/m²)
    #[serde(default)]
    pub impact_strength_z: Option<f64>,
}

/// Temperature settings for the filament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureSettings {
    /// Nozzle temperature settings
    pub nozzle: NozzleTemperature,

    /// Bed temperature settings
    #[serde(default)]
    pub bed: Option<BedTemperature>,

    /// Chamber temperature settings
    #[serde(default)]
    pub chamber: Option<ChamberTemperature>,
}

/// Nozzle temperature configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NozzleTemperature {
    /// Minimum temperature (°C)
    #[serde(default)]
    pub min: Option<f64>,

    /// Maximum temperature (°C)
    #[serde(default)]
    pub max: Option<f64>,

    /// Default temperature (°C)
    pub default: f64,

    /// First layer temperature (°C)
    #[serde(default)]
    pub first_layer: Option<f64>,
}

/// Bed temperature configuration per surface type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BedTemperature {
    #[serde(default)]
    pub default: Option<f64>,
    #[serde(default)]
    pub first_layer: Option<f64>,
    #[serde(default)]
    pub smooth_pei: Option<f64>,
    #[serde(default)]
    pub textured_pei: Option<f64>,
    #[serde(default)]
    pub engineering_plate: Option<f64>,
    #[serde(default)]
    pub glass: Option<f64>,
}

/// Chamber temperature configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChamberTemperature {
    #[serde(default)]
    pub recommended: Option<f64>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub required: bool,
}

/// Flow settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowSettings {
    /// Flow ratio multiplier
    #[serde(default = "default_1_0")]
    pub ratio: f64,

    /// Maximum volumetric speed (mm³/s)
    #[serde(default)]
    pub max_volumetric_speed: Option<f64>,

    /// Flushing volumetric speed (mm³/s)
    #[serde(default)]
    pub flush_volumetric_speed: Option<f64>,
}

/// Filament-specific cooling settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilamentCooling {
    #[serde(default)]
    pub fan_min_speed: Option<f64>,
    #[serde(default)]
    pub fan_max_speed: Option<f64>,
    #[serde(default)]
    pub fan_below_layer_time: Option<f64>,
    #[serde(default)]
    pub disable_fan_first_layers: Option<u32>,
    #[serde(default)]
    pub full_fan_speed_layer: Option<u32>,
    #[serde(default)]
    pub slow_down_layer_time: Option<f64>,
    #[serde(default)]
    pub slow_down_min_speed: Option<f64>,
    #[serde(default)]
    pub overhang_fan_speed: Option<f64>,
    #[serde(default)]
    pub overhang_fan_threshold: Option<f64>,
}

/// Multi-material settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiMaterialSettings {
    #[serde(default)]
    pub prime_volume: Option<f64>,
    #[serde(default)]
    pub minimal_purge_on_wipe_tower: Option<f64>,
    #[serde(default)]
    pub ramming_volumetric_speed: Option<f64>,
    #[serde(default)]
    pub flush_temperature: Option<f64>,
}

/// Support material properties.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SupportMaterialProperties {
    #[serde(default)]
    pub is_support_material: bool,
    #[serde(default)]
    pub soluble: bool,
    #[serde(default)]
    pub solvent: Option<String>,
    #[serde(default)]
    pub compatible_materials: Vec<String>,
}

/// Special material properties.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpecialProperties {
    #[serde(default)]
    pub translucent: bool,
    #[serde(default)]
    pub glow_in_dark: bool,
    #[serde(default)]
    pub metallic: bool,
    #[serde(default)]
    pub silk: bool,
    #[serde(default)]
    pub matte: bool,
    #[serde(default)]
    pub wood_filled: bool,
    #[serde(default)]
    pub carbon_fiber: bool,
    #[serde(default)]
    pub glass_fiber: bool,
    #[serde(default)]
    pub high_speed: bool,
    #[serde(default)]
    pub requires_enclosure: bool,
    #[serde(default)]
    pub requires_hardened_nozzle: bool,
    #[serde(default)]
    pub requires_drying: bool,
    #[serde(default)]
    pub drying_temperature: Option<f64>,
    #[serde(default)]
    pub drying_time: Option<f64>,
}

/// Environmental considerations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    #[serde(default)]
    pub requires_ventilation: bool,
    #[serde(default)]
    pub requires_air_filtration: bool,
    #[serde(default)]
    pub voc_emissions: Option<String>,
    #[serde(default)]
    pub food_safe: bool,
    #[serde(default)]
    pub biodegradable: bool,
}

/// Cost information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostInfo {
    #[serde(default)]
    pub price_per_kg: Option<f64>,
    #[serde(default = "default_currency")]
    pub currency: String,
}

/// Filament-specific G-code.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilamentGCode {
    #[serde(default)]
    pub start_gcode: Option<String>,
    #[serde(default)]
    pub end_gcode: Option<String>,
}

// ============================================================================
// Default value functions
// ============================================================================

fn default_technology() -> String {
    "FFF".to_string()
}

fn default_origin() -> String {
    "corner".to_string()
}

fn default_nozzle_diameter() -> f64 {
    0.4
}

fn default_filament_diameter() -> f64 {
    1.75
}

fn default_extruder_count() -> u32 {
    1
}

fn default_fan_count() -> u32 {
    1
}

fn default_100() -> u32 {
    100
}

fn default_100_f64() -> f64 {
    100.0
}

fn default_1_0() -> f64 {
    1.0
}

fn default_true() -> bool {
    true
}

fn default_currency() -> String {
    "USD".to_string()
}

// ============================================================================
// Implementation
// ============================================================================

impl PrinterProfile {
    /// Load a printer profile from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> ProfileResult<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Parse a printer profile from JSON string.
    pub fn from_json(json: &str) -> ProfileResult<Self> {
        let profile: Self = serde_json::from_str(json)?;
        profile.validate()?;
        Ok(profile)
    }

    /// Validate the profile.
    pub fn validate(&self) -> ProfileResult<()> {
        if self.id.is_empty() {
            return Err(ProfileError::Invalid("id cannot be empty".to_string()));
        }
        if self.build_volume.x <= 0.0 || self.build_volume.y <= 0.0 || self.build_volume.z <= 0.0 {
            return Err(ProfileError::Invalid(
                "build volume dimensions must be positive".to_string(),
            ));
        }
        if self.default_nozzle_diameter <= 0.0 {
            return Err(ProfileError::Invalid(
                "default nozzle diameter must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Get the filament diameter this printer expects.
    pub fn filament_diameter(&self) -> f64 {
        self.extruder
            .as_ref()
            .map(|e| e.filament_diameter)
            .unwrap_or(1.75)
    }

    /// Check if printer has a heated bed.
    pub fn has_heated_bed(&self) -> bool {
        self.bed.as_ref().map(|b| b.heated).unwrap_or(false)
    }

    /// Check if printer is enclosed.
    pub fn is_enclosed(&self) -> bool {
        self.enclosure.as_ref().map(|e| e.enclosed).unwrap_or(false)
    }

    /// Get maximum print speed.
    pub fn max_print_speed(&self) -> Option<f64> {
        self.limits
            .as_ref()
            .and_then(|l| l.max_speed.as_ref())
            .and_then(|s| s.print)
    }

    /// Get retraction length for this printer (from default nozzle config or global).
    pub fn retraction_length(&self) -> f64 {
        // Try to get from default nozzle config first
        let default_key = format!("{}", self.default_nozzle_diameter);
        if let Some(nozzle_config) = self.nozzle_configs.get(&default_key) {
            if let Some(retraction) = &nozzle_config.retraction {
                if let Some(length) = retraction.length {
                    return length;
                }
            }
        }
        // Fall back to global retraction
        self.retraction
            .as_ref()
            .and_then(|r| r.length)
            .unwrap_or(0.8)
    }

    /// Get nozzle config for a specific diameter.
    pub fn get_nozzle_config(&self, diameter: f64) -> Option<&NozzleConfig> {
        let key = format!("{}", diameter);
        self.nozzle_configs.get(&key)
    }

    /// Get the default nozzle config.
    pub fn default_nozzle_config(&self) -> Option<&NozzleConfig> {
        self.get_nozzle_config(self.default_nozzle_diameter)
    }

    /// Get effective nozzle diameter (for backward compatibility).
    pub fn nozzle_diameter(&self) -> f64 {
        self.nozzle
            .as_ref()
            .map(|n| n.diameter)
            .unwrap_or(self.default_nozzle_diameter)
    }
}

impl FilamentProfile {
    /// Load a filament profile from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> ProfileResult<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Parse a filament profile from a JSON string.
    pub fn from_json(json: &str) -> ProfileResult<Self> {
        let profile: Self = serde_json::from_str(json)?;
        profile.validate()?;
        Ok(profile)
    }

    /// Validate the profile.
    pub fn validate(&self) -> ProfileResult<()> {
        if self.id.is_empty() {
            return Err(ProfileError::Invalid("id cannot be empty".to_string()));
        }
        if self.brand.is_empty() {
            return Err(ProfileError::Invalid("brand cannot be empty".to_string()));
        }
        if self.material.is_empty() {
            return Err(ProfileError::Invalid(
                "material cannot be empty".to_string(),
            ));
        }
        if self.temperatures.nozzle.default <= 0.0 {
            return Err(ProfileError::Invalid(
                "default nozzle temperature must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Get the default nozzle temperature.
    pub fn nozzle_temperature(&self) -> f64 {
        self.temperatures.nozzle.default
    }

    /// Get the first layer nozzle temperature.
    pub fn first_layer_nozzle_temperature(&self) -> f64 {
        self.temperatures
            .nozzle
            .first_layer
            .unwrap_or(self.temperatures.nozzle.default)
    }

    /// Get the default bed temperature.
    pub fn bed_temperature(&self) -> Option<f64> {
        self.temperatures.bed.as_ref().and_then(|b| b.default)
    }

    /// Get the flow ratio.
    pub fn flow_ratio(&self) -> f64 {
        self.flow.as_ref().map(|f| f.ratio).unwrap_or(1.0)
    }

    /// Get the maximum volumetric speed.
    pub fn max_volumetric_speed(&self) -> Option<f64> {
        self.flow.as_ref().and_then(|f| f.max_volumetric_speed)
    }

    /// Check if this filament requires an enclosure.
    pub fn requires_enclosure(&self) -> bool {
        self.special_properties
            .as_ref()
            .map(|p| p.requires_enclosure)
            .unwrap_or(false)
    }

    /// Check if this filament requires a hardened nozzle.
    pub fn requires_hardened_nozzle(&self) -> bool {
        self.special_properties
            .as_ref()
            .map(|p| p.requires_hardened_nozzle)
            .unwrap_or(false)
    }

    /// Check if this is a support material.
    pub fn is_support_material(&self) -> bool {
        self.support_material
            .as_ref()
            .map(|s| s.is_support_material)
            .unwrap_or(false)
    }

    /// Get the material density.
    pub fn density(&self) -> Option<f64> {
        self.physical_properties.as_ref().and_then(|p| p.density)
    }

    /// Get the filament diameter.
    pub fn diameter(&self) -> f64 {
        self.physical_properties
            .as_ref()
            .map(|p| p.diameter)
            .unwrap_or(1.75)
    }
}

// ============================================================================
// Profile Registry
// ============================================================================

/// A registry for managing loaded printer and filament profiles.
#[derive(Debug, Default)]
pub struct ProfileRegistry {
    printers: HashMap<String, PrinterProfile>,
    filaments: HashMap<String, FilamentProfile>,
}

impl ProfileRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load all profiles from a directory structure.
    pub fn load_from_directory<P: AsRef<Path>>(data_dir: P) -> ProfileResult<Self> {
        let data_dir = data_dir.as_ref();
        let mut registry = Self::new();

        // Load printers
        let printers_dir = data_dir.join("printers");
        if printers_dir.exists() {
            for entry in fs::read_dir(&printers_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    match PrinterProfile::from_file(&path) {
                        Ok(profile) => {
                            registry.printers.insert(profile.id.clone(), profile);
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load printer profile {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        // Load filaments
        let filaments_dir = data_dir.join("filaments");
        if filaments_dir.exists() {
            for entry in fs::read_dir(&filaments_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    match FilamentProfile::from_file(&path) {
                        Ok(profile) => {
                            registry.filaments.insert(profile.id.clone(), profile);
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load filament profile {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        Ok(registry)
    }

    /// Add a printer profile to the registry.
    pub fn add_printer(&mut self, profile: PrinterProfile) {
        self.printers.insert(profile.id.clone(), profile);
    }

    /// Add a filament profile to the registry.
    pub fn add_filament(&mut self, profile: FilamentProfile) {
        self.filaments.insert(profile.id.clone(), profile);
    }

    /// Get a printer profile by ID.
    pub fn get_printer(&self, id: &str) -> Option<&PrinterProfile> {
        self.printers.get(id)
    }

    /// Get a filament profile by ID.
    pub fn get_filament(&self, id: &str) -> Option<&FilamentProfile> {
        self.filaments.get(id)
    }

    /// List all printer profile IDs.
    pub fn printer_ids(&self) -> impl Iterator<Item = &str> {
        self.printers.keys().map(|s| s.as_str())
    }

    /// List all filament profile IDs.
    pub fn filament_ids(&self) -> impl Iterator<Item = &str> {
        self.filaments.keys().map(|s| s.as_str())
    }

    /// Get the number of loaded printer profiles.
    pub fn printer_count(&self) -> usize {
        self.printers.len()
    }

    /// Get the number of loaded filament profiles.
    pub fn filament_count(&self) -> usize {
        self.filaments.len()
    }

    /// Find filaments compatible with a given printer.
    pub fn compatible_filaments(&self, printer_id: &str) -> Vec<&FilamentProfile> {
        self.filaments
            .values()
            .filter(|f| f.compatible_printers.contains(&printer_id.to_string()))
            .collect()
    }

    /// Find printers compatible with a given filament.
    pub fn compatible_printers(&self, filament_id: &str) -> Vec<&PrinterProfile> {
        self.printers
            .values()
            .filter(|p| p.compatible_filaments.contains(&filament_id.to_string()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_printer_profile_from_json() {
        let json = r#"{
            "id": "test-printer-0.4",
            "brand": "Test",
            "model": "Printer",
            "build_volume": { "x": 200, "y": 200, "z": 200 },
            "default_nozzle_diameter": 0.4,
            "nozzle_configs": {
                "0.4": { "diameter": 0.4 }
            }
        }"#;

        let profile = PrinterProfile::from_json(json).unwrap();
        assert_eq!(profile.id, "test-printer-0.4");
        assert_eq!(profile.brand, "Test");
        assert_eq!(profile.model, "Printer");
        assert_eq!(profile.build_volume.x, 200.0);
        assert_eq!(profile.default_nozzle_diameter, 0.4);
    }

    #[test]
    fn test_filament_profile_from_json() {
        let json = r#"{
            "id": "test-pla",
            "brand": "Test",
            "material": "PLA",
            "temperatures": {
                "nozzle": { "default": 200 }
            }
        }"#;

        let profile = FilamentProfile::from_json(json).unwrap();
        assert_eq!(profile.id, "test-pla");
        assert_eq!(profile.brand, "Test");
        assert_eq!(profile.material, "PLA");
        assert_eq!(profile.nozzle_temperature(), 200.0);
    }

    #[test]
    fn test_printer_validation() {
        let json = r#"{
            "id": "",
            "brand": "Test",
            "model": "Printer",
            "build_volume": { "x": 200, "y": 200, "z": 200 },
            "nozzle": { "diameter": 0.4 }
        }"#;

        let result = PrinterProfile::from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_filament_validation() {
        let json = r#"{
            "id": "test",
            "brand": "Test",
            "material": "PLA",
            "temperatures": {
                "nozzle": { "default": -10 }
            }
        }"#;

        let result = FilamentProfile::from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_registry() {
        let mut registry = ProfileRegistry::new();

        let printer_json = r#"{
            "id": "test-printer-0.4",
            "brand": "Test",
            "model": "Printer",
            "build_volume": { "x": 200, "y": 200, "z": 200 },
            "nozzle": { "diameter": 0.4 },
            "compatible_filaments": ["test-pla"]
        }"#;

        let filament_json = r#"{
            "id": "test-pla",
            "brand": "Test",
            "material": "PLA",
            "temperatures": {
                "nozzle": { "default": 200 }
            },
            "compatible_printers": ["test-printer-0.4"]
        }"#;

        registry.add_printer(PrinterProfile::from_json(printer_json).unwrap());
        registry.add_filament(FilamentProfile::from_json(filament_json).unwrap());

        assert_eq!(registry.printer_count(), 1);
        assert_eq!(registry.filament_count(), 1);

        let printer = registry.get_printer("test-printer-0.4").unwrap();
        assert_eq!(printer.model, "Printer");

        let filament = registry.get_filament("test-pla").unwrap();
        assert_eq!(filament.material, "PLA");

        let compatible = registry.compatible_filaments("test-printer-0.4");
        assert_eq!(compatible.len(), 1);
        assert_eq!(compatible[0].id, "test-pla");
    }

    #[test]
    fn test_load_bambu_x1_carbon_profile() {
        // Test loading actual Bambu Lab X1 Carbon profile from data directory
        let json = include_str!("../../../data/printers/bambu-lab-x1-carbon.json");
        let profile = PrinterProfile::from_json(json).unwrap();

        assert_eq!(profile.id, "bambu-lab-x1-carbon");
        assert_eq!(profile.brand, "bambu-lab");
        assert_eq!(profile.model, "X1 Carbon");
        assert_eq!(profile.build_volume.x, 256.0);
        assert_eq!(profile.build_volume.y, 256.0);
        // Z may vary slightly based on profile
        assert!(profile.is_enclosed());
        assert!(profile.has_heated_bed());

        // Check nozzle configs
        assert!(profile.nozzle_configs.len() > 0, "Expected nozzle configs");
        assert_eq!(profile.default_nozzle_diameter, 0.4);

        // Check features
        if let Some(features) = profile.features.as_ref() {
            assert!(features.auto_bed_leveling);
        }
    }

    #[test]
    fn test_load_bambu_a1_mini_profile() {
        let json = include_str!("../../../data/printers/bambu-lab-a1-mini.json");
        let profile = PrinterProfile::from_json(json).unwrap();

        assert_eq!(profile.id, "bambu-lab-a1-mini");
        assert!(profile.model.contains("A1")); // Model name may vary
        assert_eq!(profile.build_volume.x, 180.0);
        assert!(!profile.is_enclosed()); // A1 mini is not enclosed
        assert_eq!(profile.default_nozzle_diameter, 0.4);
    }

    #[test]
    fn test_load_generic_pla_profile() {
        let json = include_str!("../../../data/filaments/generic-pla-high-speed.json");
        let profile = FilamentProfile::from_json(json).unwrap();

        assert_eq!(profile.id, "generic-pla-high-speed");
        assert_eq!(profile.material, "PLA");
        assert!(!profile.requires_enclosure());
        assert!(!profile.requires_hardened_nozzle());
        assert_eq!(profile.diameter(), 1.75);
    }

    #[test]
    fn test_load_bambu_abs_profile() {
        let json = include_str!("../../../data/filaments/bambu-abs.json");
        let profile = FilamentProfile::from_json(json).unwrap();

        assert_eq!(profile.id, "bambu-abs");
        assert_eq!(profile.material, "ABS");
        assert!(profile.requires_enclosure());
        assert!(!profile.requires_hardened_nozzle());
    }

    #[test]
    fn test_load_bambu_pla_matte_profile() {
        let json = include_str!("../../../data/filaments/bambu-pla-matte.json");
        let profile = FilamentProfile::from_json(json).unwrap();

        assert_eq!(profile.id, "bambu-pla-matte");
        assert_eq!(profile.material, "PLA");

        // Check special properties
        if let Some(props) = profile.special_properties.as_ref() {
            assert!(props.matte);
        }
    }

    #[test]
    fn test_load_all_data_profiles() {
        // Load and validate all profiles from the data directory
        let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");

        let registry = ProfileRegistry::load_from_directory(&data_dir).unwrap();

        // We should have loaded our test profiles
        assert!(
            registry.printer_count() >= 2,
            "Expected at least 2 printers, got {}",
            registry.printer_count()
        );
        assert!(
            registry.filament_count() >= 3,
            "Expected at least 3 filaments, got {}",
            registry.filament_count()
        );

        // Verify we can look them up (using new profile names)
        assert!(registry.get_printer("bambu-lab-x1-carbon").is_some());
        assert!(registry.get_printer("bambu-lab-a1-mini").is_some());
        assert!(registry.get_filament("generic-pla-high-speed").is_some());
        assert!(registry.get_filament("bambu-abs").is_some());
        assert!(registry.get_filament("bambu-pla-matte").is_some());
    }
}
