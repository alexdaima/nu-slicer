import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Get outer-wall segments and check for gaps
const outerWallSegments = parsed.segments.filter(s => s.featureType === 'outer-wall' && s.type === 'extrude');

// Group by layer
const byLayer = new Map<number, typeof outerWallSegments>();
for (const seg of outerWallSegments) {
  const z = Math.round(seg.start.z * 1000) / 1000;
  if (!byLayer.has(z)) byLayer.set(z, []);
  byLayer.get(z)!.push(seg);
}

// Check first few layers for connectivity
let totalGaps = 0;
let totalConnections = 0;

for (const [z, segments] of byLayer) {
  // Sort by segment ID to maintain order
  segments.sort((a, b) => a.id - b.id);
  
  let layerGaps = 0;
  for (let i = 1; i < segments.length; i++) {
    const prev = segments[i-1].end;
    const curr = segments[i].start;
    const dist = Math.sqrt(
      Math.pow(prev.x - curr.x, 2) + 
      Math.pow(prev.y - curr.y, 2) + 
      Math.pow(prev.z - curr.z, 2)
    );
    
    if (dist > 0.01) { // Gap larger than 0.01mm
      layerGaps++;
    }
    totalConnections++;
  }
  
  totalGaps += layerGaps;
  if (z < 5) { // Print first few layers
    console.log(`Layer Z=${z}: ${segments.length} segments, ${layerGaps} gaps`);
  }
}

console.log(`\nTotal outer-wall segments: ${outerWallSegments.length}`);
console.log(`Total connections checked: ${totalConnections}`);
console.log(`Total gaps found: ${totalGaps}`);
console.log(`Gap percentage: ${(totalGaps / totalConnections * 100).toFixed(2)}%`);
