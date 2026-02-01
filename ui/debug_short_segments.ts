import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Find all segments that are very short (could create rings)
const shortSegments = parsed.segments.filter(s => {
  const dist = Math.sqrt(
    (s.end.x - s.start.x) ** 2 +
    (s.end.y - s.start.y) ** 2 +
    (s.end.z - s.start.z) ** 2
  );
  return dist < 0.02 && dist > 0; // Very short but not zero
});

console.log(`Found ${shortSegments.length} very short segments (<0.02mm)\n`);

// Group by layer
const byLayer = new Map<number, typeof shortSegments>();
for (const seg of shortSegments) {
  const z = Math.round(seg.start.z * 100) / 100;
  if (!byLayer.has(z)) byLayer.set(z, []);
  byLayer.get(z)!.push(seg);
}

// Show first few layers
const layerZs = Array.from(byLayer.keys()).sort((a, b) => a - b);
for (let i = 0; i < Math.min(3, layerZs.length); i++) {
  const z = layerZs[i];
  const segs = byLayer.get(z)!;
  console.log(`Layer Z=${z}: ${segs.length} short segments`);
  
  // Show first 5
  for (let j = 0; j < Math.min(5, segs.length); j++) {
    const s = segs[j];
    const dist = Math.sqrt(
      (s.end.x - s.start.x) ** 2 +
      (s.end.y - s.start.y) ** 2
    );
    console.log(`  Segment ${s.id} (line ${s.lineNumber}, ${s.type}): ${dist.toFixed(4)}mm`);
    console.log(`    Start: X=${s.start.x.toFixed(3)} Y=${s.start.y.toFixed(3)}`);
    console.log(`    End:   X=${s.end.x.toFixed(3)} Y=${s.end.y.toFixed(3)}`);
  }
  console.log();
}

// Look for overlapping segments (same position)
console.log('=== Checking for duplicate positions ===');
const positionMap = new Map<string, number[]>();

for (const seg of parsed.segments) {
  const startKey = `${seg.start.x.toFixed(2)},${seg.start.y.toFixed(2)},${seg.start.z.toFixed(2)}`;
  const endKey = `${seg.end.x.toFixed(2)},${seg.end.y.toFixed(2)},${seg.end.z.toFixed(2)}`;
  
  if (!positionMap.has(startKey)) positionMap.set(startKey, []);
  if (!positionMap.has(endKey)) positionMap.set(endKey, []);
  
  positionMap.get(startKey)!.push(seg.id);
  positionMap.get(endKey)!.push(seg.id);
}

// Find positions with many segments
let duplicateCount = 0;
for (const [pos, ids] of positionMap) {
  if (ids.length > 2) {
    duplicateCount++;
    if (duplicateCount <= 5) {
      console.log(`Position ${pos} has ${ids.length} segments: ${ids.slice(0, 10).join(', ')}`);
    }
  }
}
console.log(`\nTotal positions with >2 segments: ${duplicateCount}`);
