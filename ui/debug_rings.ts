import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Look for arc segments (G2/G3) in outer-wall
const outerWallArcs = parsed.segments.filter(s => 
  s.featureType === 'outer-wall' && 
  (s.type === 'extrude' || s.type === 'travel')
);

// Find arc segments that might be creating rings
// Look at layer 5 (Z around 1.0)
const layerZ = 1.20;
const layerSegments = outerWallArcs.filter(s => 
  Math.abs(s.start.z - layerZ) < 0.001
);

console.log(`Layer Z=${layerZ}: ${layerSegments.length} outer-wall segments\n`);

// Look for patterns - segments that start and end at nearly the same position
const potentialRings = layerSegments.filter((s, i) => {
  const dist = Math.sqrt(
    (s.end.x - s.start.x) ** 2 +
    (s.end.y - s.start.y) ** 2
  );
  // Very short segments that aren't part of continuous path
  return dist < 0.01 && dist > 0.001;
});

console.log(`Found ${potentialRings.length} very short segments (<0.01mm):\n`);

// Show first 10
for (let i = 0; i < Math.min(10, potentialRings.length); i++) {
  const s = potentialRings[i];
  const dist = Math.sqrt(
    (s.end.x - s.start.x) ** 2 +
    (s.end.y - s.start.y) ** 2
  );
  console.log(`Segment ${s.id} (line ${s.lineNumber}):`);
  console.log(`  Type: ${s.type}, Feature: ${s.featureType}`);
  console.log(`  Length: ${dist.toFixed(4)}mm`);
  console.log(`  Start: X=${s.start.x.toFixed(3)} Y=${s.start.y.toFixed(3)}`);
  console.log(`  End:   X=${s.end.x.toFixed(3)} Y=${s.end.y.toFixed(3)}`);
  console.log();
}

// Check for segments with large position jumps (might indicate wrong arc interpolation)
console.log(`\nChecking for unusual position changes...`);
for (let i = 1; i < layerSegments.length; i++) {
  const prev = layerSegments[i-1];
  const curr = layerSegments[i];
  const dist = Math.sqrt(
    (prev.end.x - curr.start.x) ** 2 +
    (prev.end.y - curr.start.y) ** 2
  );
  
  // Check if there's a gap
  if (dist > 0.1) {
    console.log(`\nGap of ${dist.toFixed(3)}mm between segments ${prev.id} and ${curr.id}`);
    console.log(`  Prev end: X=${prev.end.x.toFixed(3)} Y=${prev.end.y.toFixed(3)}`);
    console.log(`  Curr start: X=${curr.start.x.toFixed(3)} Y=${curr.start.y.toFixed(3)}`);
    
    // Look at the G-code
    const lines = gcodeText.split(/\r?\n/);
    console.log(`  G-code lines ${prev.lineNumber}-${curr.lineNumber}:`);
    for (let j = prev.lineNumber; j <= curr.lineNumber && j <= lines.length; j++) {
      console.log(`    ${j}: ${lines[j-1]}`);
    }
  }
}
