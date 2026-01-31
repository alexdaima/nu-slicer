//! G-code generator.
//!
//! This module provides the GCode type representing generated G-code output,
//! mirroring BambuStudio's GCode class.

use crate::{Error, Result};
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Represents generated G-code output.
///
/// This is the result of the slicing process - a complete G-code file
/// that can be written to disk or sent to a printer.
#[derive(Clone, Default)]
pub struct GCode {
    /// The G-code content as a string.
    content: String,

    /// Statistics about the generated G-code.
    pub stats: GCodeStats,
}

impl GCode {
    /// Create a new empty GCode.
    pub fn new() -> Self {
        Self {
            content: String::new(),
            stats: GCodeStats::default(),
        }
    }

    /// Create a GCode from a string.
    pub fn from_string(content: String) -> Self {
        Self {
            content,
            stats: GCodeStats::default(),
        }
    }

    /// Get the G-code content as a string.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get the G-code content as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.content.as_bytes()
    }

    /// Get the length of the G-code content in bytes.
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if the G-code is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Append a line to the G-code.
    pub fn append_line(&mut self, line: &str) {
        self.content.push_str(line);
        self.content.push('\n');
    }

    /// Append raw content to the G-code.
    pub fn append(&mut self, content: &str) {
        self.content.push_str(content);
    }

    /// Append a comment to the G-code.
    pub fn append_comment(&mut self, comment: &str) {
        self.content.push_str("; ");
        self.content.push_str(comment);
        self.content.push('\n');
    }

    /// Write the G-code to a file.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| Error::Io(e))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(self.content.as_bytes())
            .map_err(|e| Error::Io(e))?;
        writer.flush().map_err(|e| Error::Io(e))?;
        Ok(())
    }

    /// Read G-code from a file.
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::Io(e))?;
        Ok(Self::from_string(content))
    }

    /// Get the number of lines in the G-code.
    pub fn line_count(&self) -> usize {
        self.content.lines().count()
    }

    /// Iterate over the lines of the G-code.
    pub fn lines(&self) -> impl Iterator<Item = &str> {
        self.content.lines()
    }

    /// Clear the G-code content.
    pub fn clear(&mut self) {
        self.content.clear();
        self.stats = GCodeStats::default();
    }
}

impl fmt::Debug for GCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GCode({} bytes, {} lines)",
            self.len(),
            self.line_count()
        )
    }
}

impl fmt::Display for GCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

impl From<String> for GCode {
    fn from(content: String) -> Self {
        Self::from_string(content)
    }
}

impl From<GCode> for String {
    fn from(gcode: GCode) -> Self {
        gcode.content
    }
}

/// Statistics about generated G-code.
#[derive(Clone, Debug, Default)]
pub struct GCodeStats {
    /// Total number of layers.
    pub layer_count: usize,

    /// Total estimated print time (seconds).
    pub print_time_seconds: f64,

    /// Total filament used (mm).
    pub filament_used_mm: f64,

    /// Total filament used (grams, estimated).
    pub filament_used_grams: f64,

    /// Total travel distance (mm).
    pub travel_distance_mm: f64,

    /// Total extrusion distance (mm).
    pub extrusion_distance_mm: f64,

    /// Number of retractions.
    pub retraction_count: usize,

    /// Number of tool changes.
    pub tool_change_count: usize,
}

impl GCodeStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get print time formatted as HH:MM:SS.
    pub fn print_time_formatted(&self) -> String {
        let total_seconds = self.print_time_seconds as u64;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }

    /// Get filament used in meters.
    pub fn filament_used_meters(&self) -> f64 {
        self.filament_used_mm / 1000.0
    }
}

impl fmt::Display for GCodeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GCodeStats(layers={}, time={}, filament={:.2}m)",
            self.layer_count,
            self.print_time_formatted(),
            self.filament_used_meters()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcode_new() {
        let gcode = GCode::new();
        assert!(gcode.is_empty());
        assert_eq!(gcode.len(), 0);
    }

    #[test]
    fn test_gcode_append_line() {
        let mut gcode = GCode::new();
        gcode.append_line("G28");
        gcode.append_line("G1 X10 Y10");

        assert!(!gcode.is_empty());
        assert_eq!(gcode.line_count(), 2);
        assert!(gcode.content().contains("G28"));
        assert!(gcode.content().contains("G1 X10 Y10"));
    }

    #[test]
    fn test_gcode_append_comment() {
        let mut gcode = GCode::new();
        gcode.append_comment("This is a comment");

        assert!(gcode.content().starts_with("; This is a comment"));
    }

    #[test]
    fn test_gcode_from_string() {
        let content = "G28\nG1 X10\n".to_string();
        let gcode = GCode::from_string(content.clone());

        assert_eq!(gcode.content(), content);
        assert_eq!(gcode.line_count(), 2);
    }

    #[test]
    fn test_gcode_lines_iterator() {
        let mut gcode = GCode::new();
        gcode.append_line("G28");
        gcode.append_line("G1 X10");
        gcode.append_line("G1 Y20");

        let lines: Vec<&str> = gcode.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "G28");
        assert_eq!(lines[1], "G1 X10");
        assert_eq!(lines[2], "G1 Y20");
    }

    #[test]
    fn test_gcode_clear() {
        let mut gcode = GCode::new();
        gcode.append_line("G28");
        assert!(!gcode.is_empty());

        gcode.clear();
        assert!(gcode.is_empty());
    }

    #[test]
    fn test_gcode_stats_print_time_formatted() {
        let mut stats = GCodeStats::new();
        stats.print_time_seconds = 3661.0; // 1 hour, 1 minute, 1 second

        assert_eq!(stats.print_time_formatted(), "01:01:01");
    }

    #[test]
    fn test_gcode_stats_filament_meters() {
        let mut stats = GCodeStats::new();
        stats.filament_used_mm = 5000.0;

        assert!((stats.filament_used_meters() - 5.0).abs() < 1e-6);
    }
}
