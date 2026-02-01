import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Get outer-wall segments
const outerWallSegments = parsed.segments.filter(s => s.featureType === 'outer-wall' && s.type === 'extrude');

// Check a specific layer (layer 10)
const layerZ = Array.from(parsed.layers.keys()).sort((a, b) => a - b)[10];
const layerSegments = parsed.layers.get(layerZ) || [];
const outerWallInLayer = layerSegments.filter(s => s.featureType === 'outer-wall');

console.log(`Layer Z=${layerZ}:`);
console.log(`  Total segments: ${layerSegments.length}`);
console.log(`  Outer wall segments: ${outerWallInLayer.length}`);

// Check if they're consecutive
if (outerWallInLayer.length > 1) {
  console.log('\n  Outer wall segment connections:');
  for (let i = 1; i < Math.min(outerWallInLayer.length, 6); i++) {
    const prev = outerWallInLayer[i-1].end;
    const curr = outerWallInLayer[i].start;
    const dist = Math.sqrt(
      Math.pow(prev.x - curr.x, 2) + 
      Math.pow(prev.y - curr.y, 2)
    );
    const hasGap = dist > 0.01;
    console.log(`    ${i}: gap=${dist.toFixed(3)}mm ${hasGap ? 'GAP!' : 'connected'}`);
  }
}

// Count total gaps in outer wall
totalGaps = 0;
totalChecks = 0;
const byLayer = new Map<number, typeof outerWallSegments>();
for (const seg of outerWallSegments) {
  const z = Math.round(seg.start.z * 100) / 100;
  if (!byLayer.has(z)) byLayer.set(z, []);
  byLayer.get(z)!.push(seg);
}

for (const [z, segments] of byLayer) {
  segments.sort((a, b) => a.id - b.id);
  for (let i = 1; i < segments.length; i++) {
    const prev = segments[i-1].end;
    const curr = segments[i].start;
    const dist = Math.sqrt(Math.pow(prev.x - curr.x, 2) + Math.pow(prev.y - curr.y, 2));
    if (dist > 0.01) totalGaps++;
    totalChecks++;
  }
}

console.log(`\nTotal outer-wall gaps across all layers: ${totalGaps} / ${totalChecks} connections`);
console.log(`Gap percentage: ${(totalGaps / totalChecks * 100).toFixed(1)}%`);
