import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

const outerWallSegments = parsed.segments.filter(s => s.featureType === 'outer-wall');

// Test higher thresholds
for (const threshold of [0.05, 0.08, 0.1, 0.15, 0.2]) {
  const filtered = outerWallSegments.filter(s => {
    const dist = Math.sqrt(
      (s.end.x - s.start.x) ** 2 +
      (s.end.y - s.start.y) ** 2
    );
    return dist >= threshold;
  });
  
  // Count segments at problematic position
  const atPosition = filtered.filter(s => {
    const midX = (s.start.x + s.end.x) / 2;
    const midY = (s.start.y + s.end.y) / 2;
    return Math.abs(midX - 171.4) < 0.5 && Math.abs(midY - 161.2) < 0.5;
  });
  
  const removed = outerWallSegments.length - filtered.length;
  console.log(`Threshold ${threshold}mm: ${filtered.length} segments total, ${removed} removed (${(removed/outerWallSegments.length*100).toFixed(1)}%), ${atPosition.length} at pos 171.4,161.2`);
}
