//! Cooling buffer for layer time-based fan control and speed adjustments.
//!
//! This module implements cooling strategies similar to BambuStudio's CoolingBuffer:
//! - Calculates layer print time and adjusts speeds to meet minimum layer time
//! - Controls fan speed based on layer time thresholds
//! - Supports per-extruder cooling adjustments

use crate::{CoordF, ExtrusionRole};

/// Configuration for cooling behavior.
#[derive(Debug, Clone)]
pub struct CoolingConfig {
    /// Minimum layer time in seconds. Layers printing faster will be slowed down.
    pub min_layer_time: f64,
    /// Maximum layer time for full fan speed (seconds).
    pub max_layer_time: f64,
    /// Minimum print speed when slowing down for cooling (mm/s).
    pub min_print_speed: f64,
    /// Enable fan if layer time is below this threshold (seconds).
    pub fan_below_layer_time: f64,
    /// Fan speed for layers below threshold (0.0 - 1.0).
    pub fan_speed: f64,
    /// Disable fan for first N layers.
    pub disable_fan_first_layers: u32,
    /// Enable bridge fan override.
    pub bridge_fan_override: bool,
    /// Fan speed for bridges (0.0 - 1.0).
    pub bridge_fan_speed: f64,
    /// Enable overhang fan override.
    pub overhang_fan_override: bool,
    /// Fan speed for overhangs (0.0 - 1.0).
    pub overhang_fan_speed: f64,
    /// Slowdown method: proportional vs binary.
    pub slowdown_proportional: bool,
    /// Full fan speed threshold (layer time in seconds).
    pub full_fan_speed_layer_time: f64,
}

impl Default for CoolingConfig {
    fn default() -> Self {
        Self {
            min_layer_time: 5.0,
            max_layer_time: 60.0,
            min_print_speed: 10.0,
            fan_below_layer_time: 60.0,
            fan_speed: 1.0,
            disable_fan_first_layers: 1,
            bridge_fan_override: true,
            bridge_fan_speed: 1.0,
            overhang_fan_override: false,
            overhang_fan_speed: 0.5,
            slowdown_proportional: true,
            full_fan_speed_layer_time: 15.0,
        }
    }
}

impl CoolingConfig {
    /// Create a new cooling config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum layer time.
    pub fn with_min_layer_time(mut self, time: f64) -> Self {
        self.min_layer_time = time;
        self
    }

    /// Set minimum print speed.
    pub fn with_min_print_speed(mut self, speed: f64) -> Self {
        self.min_print_speed = speed;
        self
    }

    /// Set fan speed (0.0 - 1.0).
    pub fn with_fan_speed(mut self, speed: f64) -> Self {
        self.fan_speed = speed.clamp(0.0, 1.0);
        self
    }

    /// Disable fan for first N layers.
    pub fn with_disable_fan_first_layers(mut self, layers: u32) -> Self {
        self.disable_fan_first_layers = layers;
        self
    }

    /// Enable/disable bridge fan override.
    pub fn with_bridge_fan(mut self, enabled: bool, speed: f64) -> Self {
        self.bridge_fan_override = enabled;
        self.bridge_fan_speed = speed.clamp(0.0, 1.0);
        self
    }
}

/// Represents a single move/extrusion segment for cooling calculations.
#[derive(Debug, Clone)]
pub struct CoolingMove {
    /// Length of the move in mm.
    pub length: f64,
    /// Original feedrate in mm/s.
    pub feedrate: f64,
    /// Whether this is a travel move (non-extrusion).
    pub is_travel: bool,
    /// Whether this can be slowed down.
    pub can_slowdown: bool,
    /// Extrusion role for this move.
    pub role: Option<ExtrusionRole>,
    /// Time to execute this move at original speed (seconds).
    pub time: f64,
    /// Adjusted feedrate after cooling slowdown (mm/s).
    pub adjusted_feedrate: f64,
}

impl CoolingMove {
    /// Create a new cooling move.
    pub fn new(length: f64, feedrate: f64, is_travel: bool, role: Option<ExtrusionRole>) -> Self {
        let time = if feedrate > 0.0 {
            length / feedrate
        } else {
            0.0
        };
        let can_slowdown = !is_travel && role != Some(ExtrusionRole::BridgeInfill);
        Self {
            length,
            feedrate,
            is_travel,
            can_slowdown,
            role,
            time,
            adjusted_feedrate: feedrate,
        }
    }

    /// Create a travel move.
    pub fn travel(length: f64, feedrate: f64) -> Self {
        Self::new(length, feedrate, true, None)
    }

    /// Create an extrusion move.
    pub fn extrusion(length: f64, feedrate: f64, role: ExtrusionRole) -> Self {
        Self::new(length, feedrate, false, Some(role))
    }

    /// Calculate time at current adjusted feedrate.
    pub fn adjusted_time(&self) -> f64 {
        if self.adjusted_feedrate > 0.0 {
            self.length / self.adjusted_feedrate
        } else {
            0.0
        }
    }
}

/// Per-extruder adjustments for cooling.
#[derive(Debug, Clone)]
pub struct PerExtruderAdjustments {
    /// Extruder index.
    pub extruder_id: u32,
    /// All moves for this extruder.
    pub moves: Vec<CoolingMove>,
    /// Total extrusion time (excluding travels).
    pub extrusion_time: f64,
    /// Total travel time.
    pub travel_time: f64,
    /// Time that can be slowed down.
    pub slowdown_time: f64,
    /// Slowdown factor applied (1.0 = no slowdown).
    pub slowdown_factor: f64,
}

impl PerExtruderAdjustments {
    /// Create new per-extruder adjustments.
    pub fn new(extruder_id: u32) -> Self {
        Self {
            extruder_id,
            moves: Vec::new(),
            extrusion_time: 0.0,
            travel_time: 0.0,
            slowdown_time: 0.0,
            slowdown_factor: 1.0,
        }
    }

    /// Add a move to this extruder's list.
    pub fn add_move(&mut self, mov: CoolingMove) {
        if mov.is_travel {
            self.travel_time += mov.time;
        } else {
            self.extrusion_time += mov.time;
            if mov.can_slowdown {
                self.slowdown_time += mov.time;
            }
        }
        self.moves.push(mov);
    }

    /// Get total time at original speeds.
    pub fn total_time(&self) -> f64 {
        self.extrusion_time + self.travel_time
    }

    /// Get total time after adjustments.
    pub fn adjusted_total_time(&self) -> f64 {
        self.moves.iter().map(|m| m.adjusted_time()).sum()
    }

    /// Apply slowdown factor to all eligible moves.
    pub fn apply_slowdown(&mut self, factor: f64, min_speed: f64) {
        self.slowdown_factor = factor;
        for mov in &mut self.moves {
            if mov.can_slowdown {
                let new_speed = (mov.feedrate / factor).max(min_speed);
                mov.adjusted_feedrate = new_speed;
            }
        }
    }
}

/// Cooling buffer that manages layer cooling and fan control.
#[derive(Debug)]
pub struct CoolingBuffer {
    /// Cooling configuration.
    config: CoolingConfig,
}

impl CoolingBuffer {
    /// Create a new cooling buffer with the given configuration.
    pub fn new(config: CoolingConfig) -> Self {
        Self { config }
    }

    /// Create a cooling buffer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CoolingConfig::default())
    }

    /// Get the cooling configuration.
    pub fn config(&self) -> &CoolingConfig {
        &self.config
    }

    /// Calculate the slowdown factor needed to meet minimum layer time.
    ///
    /// Returns the factor by which print speeds should be divided.
    /// A factor of 1.0 means no slowdown needed.
    pub fn calculate_layer_slowdown(
        &self,
        per_extruder_adjustments: &mut [PerExtruderAdjustments],
    ) -> f64 {
        // Calculate total layer time
        let total_time: f64 = per_extruder_adjustments
            .iter()
            .map(|adj| adj.total_time())
            .sum();

        // If we're already at or above minimum layer time, no slowdown needed
        if total_time >= self.config.min_layer_time {
            return 1.0;
        }

        // Calculate how much time can be slowed down
        let total_slowdown_time: f64 = per_extruder_adjustments
            .iter()
            .map(|adj| adj.slowdown_time)
            .sum();

        let fixed_time = total_time - total_slowdown_time;

        // If nothing can be slowed down, return 1.0
        if total_slowdown_time <= 0.0 {
            return 1.0;
        }

        // Calculate required slowdown factor
        let target_slowdown_time = self.config.min_layer_time - fixed_time;
        let slowdown_factor = if target_slowdown_time > 0.0 {
            target_slowdown_time / total_slowdown_time
        } else {
            1.0
        };

        // The slowdown factor is how much longer we need the slowable time to be
        // So if we need 2x the time, we divide speed by 2 (factor = 2)
        let factor = slowdown_factor.max(1.0);

        // Calculate the maximum factor based on minimum print speed
        // We need to check all moves to find the limiting factor
        let mut max_factor = f64::MAX;
        for adj in per_extruder_adjustments.iter() {
            for mov in &adj.moves {
                if mov.can_slowdown && mov.feedrate > 0.0 {
                    let move_max_factor = mov.feedrate / self.config.min_print_speed;
                    max_factor = max_factor.min(move_max_factor);
                }
            }
        }

        // Clamp the factor
        let final_factor = factor.min(max_factor).max(1.0);

        // Apply the slowdown to all extruders
        for adj in per_extruder_adjustments.iter_mut() {
            adj.apply_slowdown(final_factor, self.config.min_print_speed);
        }

        final_factor
    }

    /// Calculate fan speed for a given layer.
    ///
    /// Returns fan speed as a value from 0.0 to 1.0.
    pub fn calculate_fan_speed(&self, layer_index: u32, layer_time: f64) -> f64 {
        // Disable fan for first layers
        if layer_index < self.config.disable_fan_first_layers {
            return 0.0;
        }

        // If layer time is above threshold, no fan needed
        if layer_time >= self.config.fan_below_layer_time {
            return 0.0;
        }

        // Interpolate fan speed based on layer time
        if layer_time <= self.config.full_fan_speed_layer_time {
            // Full fan speed for very fast layers
            self.config.fan_speed
        } else {
            // Linear interpolation between full fan and no fan
            let range = self.config.fan_below_layer_time - self.config.full_fan_speed_layer_time;
            if range > 0.0 {
                let t = (self.config.fan_below_layer_time - layer_time) / range;
                t * self.config.fan_speed
            } else {
                self.config.fan_speed
            }
        }
    }

    /// Get fan speed for bridges.
    pub fn bridge_fan_speed(&self) -> Option<f64> {
        if self.config.bridge_fan_override {
            Some(self.config.bridge_fan_speed)
        } else {
            None
        }
    }

    /// Get fan speed for overhangs.
    pub fn overhang_fan_speed(&self) -> Option<f64> {
        if self.config.overhang_fan_override {
            Some(self.config.overhang_fan_speed)
        } else {
            None
        }
    }

    /// Process a layer's moves and apply cooling adjustments.
    ///
    /// Returns the adjusted moves and the calculated fan speed.
    pub fn process_layer(
        &self,
        layer_index: u32,
        moves: Vec<CoolingMove>,
        extruder_id: u32,
    ) -> CoolingResult {
        let mut adjustments = vec![PerExtruderAdjustments::new(extruder_id)];

        for mov in moves {
            adjustments[0].add_move(mov);
        }

        let original_time = adjustments[0].total_time();
        let slowdown_factor = self.calculate_layer_slowdown(&mut adjustments);
        let adjusted_time = adjustments[0].adjusted_total_time();
        let fan_speed = self.calculate_fan_speed(layer_index, adjusted_time);

        CoolingResult {
            moves: adjustments.into_iter().next().unwrap().moves,
            original_time,
            adjusted_time,
            slowdown_factor,
            fan_speed,
        }
    }
}

impl Default for CoolingBuffer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Result of cooling processing for a layer.
#[derive(Debug, Clone)]
pub struct CoolingResult {
    /// Adjusted moves with updated feedrates.
    pub moves: Vec<CoolingMove>,
    /// Original layer time in seconds.
    pub original_time: f64,
    /// Adjusted layer time in seconds.
    pub adjusted_time: f64,
    /// Slowdown factor applied.
    pub slowdown_factor: f64,
    /// Calculated fan speed (0.0 - 1.0).
    pub fan_speed: f64,
}

impl CoolingResult {
    /// Check if any slowdown was applied.
    pub fn has_slowdown(&self) -> bool {
        self.slowdown_factor > 1.0
    }

    /// Check if fan is enabled.
    pub fn fan_enabled(&self) -> bool {
        self.fan_speed > 0.0
    }

    /// Get fan speed as percentage (0-100).
    pub fn fan_speed_percent(&self) -> u32 {
        (self.fan_speed * 100.0).round() as u32
    }
}

/// Estimate layer time from path lengths and feedrates.
pub fn estimate_layer_time(
    path_lengths: &[CoordF],
    feedrates: &[CoordF],
    travel_length: CoordF,
    travel_feedrate: CoordF,
) -> f64 {
    let extrusion_time: f64 = path_lengths
        .iter()
        .zip(feedrates.iter())
        .map(|(&len, &feed)| if feed > 0.0 { len / feed } else { 0.0 })
        .sum();

    let travel_time = if travel_feedrate > 0.0 {
        travel_length / travel_feedrate
    } else {
        0.0
    };

    extrusion_time + travel_time
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooling_config_default() {
        let config = CoolingConfig::default();
        assert_eq!(config.min_layer_time, 5.0);
        assert_eq!(config.min_print_speed, 10.0);
        assert_eq!(config.fan_speed, 1.0);
    }

    #[test]
    fn test_cooling_config_builder() {
        let config = CoolingConfig::new()
            .with_min_layer_time(10.0)
            .with_min_print_speed(15.0)
            .with_fan_speed(0.8);

        assert_eq!(config.min_layer_time, 10.0);
        assert_eq!(config.min_print_speed, 15.0);
        assert_eq!(config.fan_speed, 0.8);
    }

    #[test]
    fn test_cooling_move_creation() {
        let travel = CoolingMove::travel(10.0, 100.0);
        assert!(travel.is_travel);
        assert!(!travel.can_slowdown);
        assert!((travel.time - 0.1).abs() < 0.001);

        let extrusion = CoolingMove::extrusion(20.0, 50.0, ExtrusionRole::Perimeter);
        assert!(!extrusion.is_travel);
        assert!(extrusion.can_slowdown);
        assert!((extrusion.time - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_bridge_cannot_slowdown() {
        let bridge = CoolingMove::extrusion(10.0, 30.0, ExtrusionRole::BridgeInfill);
        assert!(!bridge.can_slowdown);
    }

    #[test]
    fn test_per_extruder_adjustments() {
        let mut adj = PerExtruderAdjustments::new(0);
        adj.add_move(CoolingMove::travel(10.0, 100.0));
        adj.add_move(CoolingMove::extrusion(20.0, 50.0, ExtrusionRole::Perimeter));

        assert!((adj.travel_time - 0.1).abs() < 0.001);
        assert!((adj.extrusion_time - 0.4).abs() < 0.001);
        assert!((adj.total_time() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_slowdown_calculation_no_slowdown_needed() {
        let config = CoolingConfig {
            min_layer_time: 5.0,
            min_print_speed: 10.0,
            ..Default::default()
        };
        let buffer = CoolingBuffer::new(config);

        // Create a layer that takes 10 seconds (above minimum)
        let mut adj = PerExtruderAdjustments::new(0);
        adj.add_move(CoolingMove::extrusion(
            500.0,
            50.0,
            ExtrusionRole::Perimeter,
        )); // 10 seconds

        let mut adjustments = vec![adj];
        let factor = buffer.calculate_layer_slowdown(&mut adjustments);

        assert!((factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slowdown_calculation_slowdown_needed() {
        let config = CoolingConfig {
            min_layer_time: 10.0,
            min_print_speed: 10.0,
            ..Default::default()
        };
        let buffer = CoolingBuffer::new(config);

        // Create a layer that takes 5 seconds (below minimum)
        let mut adj = PerExtruderAdjustments::new(0);
        adj.add_move(CoolingMove::extrusion(
            250.0,
            50.0,
            ExtrusionRole::Perimeter,
        )); // 5 seconds

        let mut adjustments = vec![adj];
        let factor = buffer.calculate_layer_slowdown(&mut adjustments);

        // Should slow down by factor of 2 to reach 10 seconds
        assert!(factor > 1.0);
        assert!((factor - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_fan_speed_first_layer_disabled() {
        let config = CoolingConfig {
            disable_fan_first_layers: 2,
            ..Default::default()
        };
        let buffer = CoolingBuffer::new(config);

        assert_eq!(buffer.calculate_fan_speed(0, 1.0), 0.0);
        assert_eq!(buffer.calculate_fan_speed(1, 1.0), 0.0);
        assert!(buffer.calculate_fan_speed(2, 1.0) > 0.0);
    }

    #[test]
    fn test_fan_speed_based_on_layer_time() {
        let config = CoolingConfig {
            disable_fan_first_layers: 0,
            fan_below_layer_time: 60.0,
            full_fan_speed_layer_time: 15.0,
            fan_speed: 1.0,
            ..Default::default()
        };
        let buffer = CoolingBuffer::new(config);

        // Above threshold - no fan
        assert_eq!(buffer.calculate_fan_speed(5, 100.0), 0.0);

        // Below full fan threshold - full fan
        assert_eq!(buffer.calculate_fan_speed(5, 10.0), 1.0);

        // In between - interpolated
        let speed = buffer.calculate_fan_speed(5, 37.5);
        assert!(speed > 0.0 && speed < 1.0);
    }

    #[test]
    fn test_process_layer() {
        let config = CoolingConfig {
            min_layer_time: 10.0,
            min_print_speed: 10.0,
            disable_fan_first_layers: 0,
            fan_below_layer_time: 60.0,
            ..Default::default()
        };
        let buffer = CoolingBuffer::new(config);

        let moves = vec![
            CoolingMove::extrusion(250.0, 50.0, ExtrusionRole::Perimeter), // 5 seconds
        ];

        let result = buffer.process_layer(5, moves, 0);

        assert!(result.has_slowdown());
        assert!(result.adjusted_time >= 10.0 - 0.1);
        assert!(result.fan_enabled());
    }

    #[test]
    fn test_bridge_fan_speed() {
        let config = CoolingConfig {
            bridge_fan_override: true,
            bridge_fan_speed: 0.8,
            ..Default::default()
        };
        let buffer = CoolingBuffer::new(config);

        assert_eq!(buffer.bridge_fan_speed(), Some(0.8));

        let config_disabled = CoolingConfig {
            bridge_fan_override: false,
            ..Default::default()
        };
        let buffer_disabled = CoolingBuffer::new(config_disabled);

        assert_eq!(buffer_disabled.bridge_fan_speed(), None);
    }

    #[test]
    fn test_cooling_result() {
        let result = CoolingResult {
            moves: vec![],
            original_time: 5.0,
            adjusted_time: 10.0,
            slowdown_factor: 2.0,
            fan_speed: 0.75,
        };

        assert!(result.has_slowdown());
        assert!(result.fan_enabled());
        assert_eq!(result.fan_speed_percent(), 75);
    }

    #[test]
    fn test_estimate_layer_time() {
        let path_lengths = vec![100.0, 200.0];
        let feedrates = vec![50.0, 100.0];
        let travel_length = 50.0;
        let travel_feedrate = 100.0;

        let time = estimate_layer_time(&path_lengths, &feedrates, travel_length, travel_feedrate);

        // 100/50 + 200/100 + 50/100 = 2 + 2 + 0.5 = 4.5 seconds
        assert!((time - 4.5).abs() < 0.001);
    }
}
