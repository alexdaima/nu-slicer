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

// Group by layer
const byLayer = new Map<number, typeof outerWallSegments>();
for (const seg of outerWallSegments) {
  const z = Math.round(seg.start.z * 100) / 100;
  if (!byLayer.has(z)) byLayer.set(z, []);
  byLayer.get(z)!.push(seg);
}

// Sort layers
const sortedLayerZs = Array.from(byLayer.keys()).sort((a, b) => a - b);

// Focus on layer Z=1.20 which had gaps
const targetLayerZ = 1.20;
const layerSegments = byLayer.get(targetLayerZ) || [];

console.log(`\n=== Detailed analysis of Layer Z=${targetLayerZ.toFixed(2)} ===`);
console.log(`Total outer-wall segments: ${layerSegments.length}`);

// Sort by ID to maintain parse order
layerSegments.sort((a, b) => a.id - b.id);

// Find and analyze gaps
console.log(`\n=== Gap Analysis ===`);
let gapCount = 0;
for (let i = 1; i < layerSegments.length; i++) {
  const prev = layerSegments[i-1];
  const curr = layerSegments[i];
  const dist = Math.sqrt(
    (prev.end.x - curr.start.x) ** 2 +
    (prev.end.y - curr.start.y) ** 2
  );
  
  if (dist > 0.01) {
    gapCount++;
    console.log(`\nGap #${gapCount} between segments ${prev.id} and ${curr.id}:`);
    console.log(`  Distance: ${dist.toFixed(4)}mm`);
    console.log(`  Previous segment end:   X=${prev.end.x.toFixed(3)} Y=${prev.end.y.toFixed(3)}`);
    console.log(`  Current segment start:  X=${curr.start.x.toFixed(3)} Y=${curr.start.y.toFixed(3)}`);
    console.log(`  Previous line #: ${prev.lineNumber}, Current line #: ${curr.lineNumber}`);
    
    // Look at the G-code around these lines
    const lines = gcodeText.split(/\r?\n/);
    const startLine = Math.max(0, prev.lineNumber - 2);
    const endLine = Math.min(lines.length, curr.lineNumber + 2);
    console.log(`  G-code context (lines ${startLine}-${endLine}):`);
    for (let j = startLine; j <= endLine; j++) {
      const prefix = j === prev.lineNumber ? 'P>' : j === curr.lineNumber ? 'C>' : '  ';
      console.log(`    ${prefix} ${j}: ${lines[j-1] || '(empty)'}`);
    }
  }
}

console.log(`\n=== Summary ===`);
console.log(`Total gaps found: ${gapCount}`);
console.log(`Total segments: ${layerSegments.length}`);
console.log(`Gap percentage: ${(gapCount / (layerSegments.length - 1) * 100).toFixed(1)}%`);
