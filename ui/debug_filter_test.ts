import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Count segments that would be filtered with different thresholds
const thresholds = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05];
const outerWallSegments = parsed.segments.filter(s => s.featureType === 'outer-wall');

console.log(`Total outer-wall segments: ${outerWallSegments.length}\n`);

for (const threshold of thresholds) {
  const filtered = outerWallSegments.filter(s => {
    const dist = Math.sqrt(
      (s.end.x - s.start.x) ** 2 +
      (s.end.y - s.start.y) ** 2 +
      (s.end.z - s.start.z) ** 2
    );
    return dist >= threshold;
  });
  
  const removed = outerWallSegments.length - filtered.length;
  console.log(`Threshold ${threshold}mm: ${filtered.length} segments, ${removed} filtered out (${(removed/outerWallSegments.length*100).toFixed(1)}%)`);
}

// Check remaining segments at the problematic position
console.log('\n=== Checking position 171.4,161.2 with 0.05mm threshold ===');
const filteredSegments = outerWallSegments.filter(s => {
  const dist = Math.sqrt(
    (s.end.x - s.start.x) ** 2 +
    (s.end.y - s.start.y) ** 2 +
    (s.end.z - s.start.z) ** 2
  );
  return dist >= 0.05;
});

const atPosition = filteredSegments.filter(s => {
  const midX = (s.start.x + s.end.x) / 2;
  const midY = (s.start.y + s.end.y) / 2;
  return Math.abs(midX - 171.4) < 0.1 && Math.abs(midY - 161.2) < 0.1;
});

console.log(`Segments at 171.4,161.2 after filtering: ${atPosition.length} (was 24)`);
