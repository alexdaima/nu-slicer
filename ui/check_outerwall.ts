const fs = require('fs');
const path = require('path');

// Read the gcode file and look for outer wall patterns
const gcodePath = path.join(__dirname, '../data/reference_gcodes/3DBenchy.gcode');
const gcodeText = fs.readFileSync(gcodePath, 'utf-8');
const lines = gcodeText.split(/\r?\n/);

let inOuterWall = false;
let outerWallCount = 0;
let outerWallWithMoves = 0;
let moveCount = 0;

for (let i = 0; i < lines.length; i++) {
  const line = lines[i].trim();
  
  if (line.includes('; FEATURE: Outer wall')) {
    inOuterWall = true;
    outerWallCount++;
    moveCount = 0;
  } else if (line.includes('; FEATURE:')) {
    if (inOuterWall && moveCount > 0) {
      outerWallWithMoves++;
    }
    inOuterWall = false;
  }
  
  if (inOuterWall && (line.startsWith('G0') || line.startsWith('G1'))) {
    const hasE = /E[\d.-]/.test(line);
    if (!hasE) {
      moveCount++;
    }
  }
}

console.log(`Outer wall feature count: ${outerWallCount}`);
console.log(`Outer walls with travel moves: ${outerWallWithMoves}`);
console.log(`Percentage with gaps: ${(outerWallWithMoves / outerWallCount * 100).toFixed(1)}%`);
