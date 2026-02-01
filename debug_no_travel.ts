import { GCodeParser } from './src/parser/gcodeParser';
import { readFileSync } from 'fs';

const gcodeText = readFileSync('../data/reference_gcodes/3DBenchy.gcode', 'utf-8');
const parser = new GCodeParser();
const parsed = parser.parse(gcodeText);

// Count what would be rendered now (only extrude moves)
const renderable = parsed.segments.filter(s => s.type === 'extrude');
const outerWallRenderable = renderable.filter(s => s.featureType === 'outer-wall');

console.log(`Total extrude segments: ${renderable.length}`);
console.log(`Outer-wall extrude segments: ${outerWallRenderable.length}`);
console.log(`\nFiltered out:`);
console.log(`  Travel moves: ${parsed.segments.filter(s => s.type === 'travel').length}`);
console.log(`  Retract moves: ${parsed.segments.filter(s => s.type === 'retract').length}`);
