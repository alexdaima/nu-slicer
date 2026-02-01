import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Get outer-wall extrusion segments
const outerWallSegments = parsed.segments.filter(s => 
  s.featureType === 'outer-wall' && s.type === 'extrude'
);

console.log(`Total outer-wall extrude segments: ${outerWallSegments.length}`);

// Group by layer and check connectivity
const byLayer = new Map<number, typeof outerWallSegments>();
for (const seg of outerWallSegments) {
  const z = Math.round(seg.start.z * 100) / 100;
  if (!byLayer.has(z)) byLayer.set(z, []);
  byLayer.get(z)!.push(seg);
}

// Check layer 5 (around Z=1.0)
const layerZ = Array.from(byLayer.keys()).sort((a, b) => a - b)[5];
const layerSegments = byLayer.get(layerZ) || [];

console.log(`\nLayer Z=${layerZ.toFixed(2)}:`);
console.log(`  Segments: ${layerSegments.length}`);

// Check if consecutive segments connect
let connected = 0;
let disconnected = 0;
for (let i = 1; i < layerSegments.length; i++) {
  const prev = layerSegments[i-1];
  const curr = layerSegments[i];
  const dist = Math.sqrt(
    (prev.end.x - curr.start.x) ** 2 +
    (prev.end.y - curr.start.y) ** 2
  );
  if (dist < 0.001) connected++;
  else disconnected++;
}
console.log(`  Connected: ${connected}, Disconnected: ${disconnected}`);

// Show first few segment connections
console.log(`\n  First 5 segment gaps:`);
for (let i = 1; i < Math.min(6, layerSegments.length); i++) {
  const prev = layerSegments[i-1];
  const curr = layerSegments[i];
  const dist = Math.sqrt(
    (prev.end.x - curr.start.x) ** 2 +
    (prev.end.y - curr.start.y) ** 2
  );
  console.log(`    ${i}: ${dist.toFixed(4)}mm`);
}
