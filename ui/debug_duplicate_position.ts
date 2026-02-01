import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Find segments at the duplicate position
const targetPos = { x: 187.00, y: 160.00, z: 5.00 };
const matchingSegments = parsed.segments.filter(s => {
  const startMatch = Math.abs(s.start.x - targetPos.x) < 0.01 && 
                     Math.abs(s.start.y - targetPos.y) < 0.01 && 
                     Math.abs(s.start.z - targetPos.z) < 0.01;
  const endMatch = Math.abs(s.end.x - targetPos.x) < 0.01 && 
                   Math.abs(s.end.y - targetPos.y) < 0.01 && 
                   Math.abs(s.end.z - targetPos.z) < 0.01;
  return startMatch || endMatch;
});

console.log(`Found ${matchingSegments.length} segments at/near position ${targetPos.x},${targetPos.y},${targetPos.z}:\n`);

// Show details
for (const s of matchingSegments.slice(0, 10)) {
  const isStart = Math.abs(s.start.x - targetPos.x) < 0.01 && 
                  Math.abs(s.start.y - targetPos.y) < 0.01;
  console.log(`Segment ${s.id} (line ${s.lineNumber}):`);
  console.log(`  Type: ${s.type}, Feature: ${s.featureType}`);
  console.log(`  Position: ${isStart ? 'START' : 'END'} of segment`);
  console.log(`  Start: X=${s.start.x.toFixed(3)} Y=${s.start.y.toFixed(3)} Z=${s.start.z.toFixed(3)}`);
  console.log(`  End:   X=${s.end.x.toFixed(3)} Y=${s.end.y.toFixed(3)} Z=${s.end.z.toFixed(3)}`);
  console.log();
}

// Look at the G-code around these lines
const lines = gcodeText.split(/\r?\n/);
const lineNumbers = matchingSegments.map(s => s.lineNumber).sort((a, b) => a - b);
const minLine = Math.min(...lineNumbers);
const maxLine = Math.max(...lineNumbers);

console.log(`G-code from line ${minLine-2} to ${maxLine+2}:`);
for (let i = minLine - 2; i <= maxLine + 2 && i < lines.length; i++) {
  if (i > 0) {
    const marker = lineNumbers.includes(i) ? '>>' : '  ';
    console.log(`${marker} ${i}: ${lines[i-1]}`);
  }
}
