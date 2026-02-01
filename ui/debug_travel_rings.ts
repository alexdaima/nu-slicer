import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Look for outer-wall travel moves (which are being rendered with thin radius)
const outerWallTravels = parsed.segments.filter(s => 
  s.featureType === 'outer-wall' && s.type === 'travel'
);

console.log(`Found ${outerWallTravels.length} outer-wall travel segments\n`);

// Group by layer and look at a specific layer
const byLayer = new Map<number, typeof outerWallTravels>();
for (const seg of outerWallTravels) {
  const z = Math.round(seg.start.z * 100) / 100;
  if (!byLayer.has(z)) byLayer.set(z, []);
  byLayer.get(z)!.push(seg);
}

// Show layer 1.2
const layerZ = 1.20;
const travels = byLayer.get(layerZ) || [];

console.log(`Layer Z=${layerZ}: ${travels.length} outer-wall travel segments\n`);

// Show first 20
for (let i = 0; i < Math.min(20, travels.length); i++) {
  const s = travels[i];
  const dist = Math.sqrt(
    (s.end.x - s.start.x) ** 2 +
    (s.end.y - s.start.y) ** 2
  );
  console.log(`Travel ${s.id} (line ${s.lineNumber}): ${dist.toFixed(3)}mm`);
  console.log(`  Start: X=${s.start.x.toFixed(3)} Y=${s.start.y.toFixed(3)}`);
  console.log(`  End:   X=${s.end.x.toFixed(3)} Y=${s.end.y.toFixed(3)}`);
  
  // Look at G-code
  const lines = gcodeText.split(/\r?\n/);
  console.log(`  G-code: ${lines[s.lineNumber-1]}`);
  console.log();
}
