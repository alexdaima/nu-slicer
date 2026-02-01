import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Find all segments around lines 6298-6310
const targetLineRange = { start: 6295, end: 6310 };

console.log(`Looking for segments with line numbers ${targetLineRange.start}-${targetLineRange.end}:\n`);

const relevantSegments = parsed.segments.filter(s => 
  s.lineNumber >= targetLineRange.start && s.lineNumber <= targetLineRange.end
);

console.log(`Found ${relevantSegments.length} segments:\n`);

for (const seg of relevantSegments) {
  console.log(`Segment ${seg.id} (line ${seg.lineNumber}):`);
  console.log(`  Type: ${seg.type}, Feature: ${seg.featureType}`);
  console.log(`  Start: X=${seg.start.x.toFixed(3)} Y=${seg.start.y.toFixed(3)} Z=${seg.start.z.toFixed(3)}`);
  console.log(`  End:   X=${seg.end.x.toFixed(3)} Y=${seg.end.y.toFixed(3)} Z=${seg.end.z.toFixed(3)}`);
  console.log(`  E: ${seg.e.toFixed(5)}, Speed: ${seg.speed}`);
  console.log();
}

// Also show the G-code lines
console.log('=== G-code lines ===');
const lines = gcodeText.split(/\r?\n/);
for (let i = targetLineRange.start - 1; i < targetLineRange.end && i < lines.length; i++) {
  console.log(`${i + 1}: ${lines[i]}`);
}
