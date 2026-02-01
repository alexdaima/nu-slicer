import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Look for outer-wall segments that might form rings
// Check for segments that start and end very close to each other (circular patterns)
const outerWallSegments = parsed.segments.filter(s => s.featureType === 'outer-wall' && s.type === 'extrude');

console.log(`Total outer-wall extrude segments: ${outerWallSegments.length}\n`);

// Look for potential circular patterns
// Segments where end is close to start (small arcs or loops)
const potentialRings = outerWallSegments.filter(s => {
  const dist = Math.sqrt(
    (s.end.x - s.start.x) ** 2 +
    (s.end.y - s.start.y) ** 2
  );
  // Very short segments that might form a ring when rendered
  return dist < 0.05 && dist > 0.001;
});

console.log(`Found ${potentialRings.length} short outer-wall segments (<0.05mm)\n`);

// Group by approximate position
const byPosition = new Map<string, number>();
for (const s of potentialRings) {
  const midX = (s.start.x + s.end.x) / 2;
  const midY = (s.start.y + s.end.y) / 2;
  const key = `${midX.toFixed(1)},${midY.toFixed(1)}`;
  byPosition.set(key, (byPosition.get(key) || 0) + 1);
}

// Show positions with multiple short segments
console.log('Positions with multiple short segments (potential rings):');
const sortedPositions = Array.from(byPosition.entries())
  .sort((a, b) => b[1] - a[1])
  .slice(0, 10);

for (const [pos, count] of sortedPositions) {
  console.log(`  ${pos}: ${count} segments`);
}

// Look at one specific case
if (potentialRings.length > 0) {
  const example = potentialRings[0];
  console.log(`\nExample short segment ${example.id} (line ${example.lineNumber}):`);
  console.log(`  Start: X=${example.start.x.toFixed(4)} Y=${example.start.y.toFixed(4)}`);
  console.log(`  End:   X=${example.end.x.toFixed(4)} Y=${example.end.y.toFixed(4)}`);
  const dist = Math.sqrt(
    (example.end.x - example.start.x) ** 2 +
    (example.end.y - example.start.y) ** 2
  );
  console.log(`  Length: ${dist.toFixed(4)}mm`);
  
  // Look at the G-code
  const lines = gcodeText.split(/\r?\n/);
  console.log(`\nG-code around line ${example.lineNumber}:`);
  for (let i = example.lineNumber - 2; i <= example.lineNumber + 2 && i < lines.length; i++) {
    if (i > 0) {
      console.log(`  ${i}: ${lines[i-1]}`);
    }
  }
}
