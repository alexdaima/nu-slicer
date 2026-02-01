// Debug script to check segment counts by feature
import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

const counts: Record<string, number> = {};
for (const seg of parsed.segments) {
  const feature = seg.featureType || 'custom';
  const type = seg.type;
  const key = `${feature} (${type})`;
  counts[key] = (counts[key] || 0) + 1;
}

console.log('Segment counts by feature and type:');
for (const [key, count] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
  console.log(`${key}: ${count.toLocaleString()}`);
}
console.log(`\nTotal segments: ${parsed.segments.length.toLocaleString()}`);
console.log(`Total layers: ${parsed.totalLayers}`);
