//! Configuration module for print settings.
//!
//! This module provides configuration types for controlling the slicing
//! and printing process, mirroring BambuStudio's PrintConfig classes.

mod print_config;
mod region_config;

pub use print_config::{
    GCodeFlavor, InfillPattern, PerimeterMode, PrintConfig, PrintObjectConfig, SeamPosition,
    SupportType,
};
pub use region_config::{FuzzySkinMode, IroningType, PrintRegionConfig};
