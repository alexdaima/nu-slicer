Default to using Bun instead of Node.js.

- Use `bun <file>` instead of `node <file>` or `ts-node <file>`
- Use `bun test` instead of `jest` or `vitest`
- Use `bun build <file.html|file.ts|file.css>` instead of `webpack` or `esbuild`
- Use `bun install` instead of `npm install` or `yarn install` or `pnpm install`
- Use `bun run <script>` instead of `npm run <script>` or `yarn run <script>` or `pnpm run <script>`
- Bun automatically loads .env, so don't use dotenv.

## APIs

- `Bun.serve()` supports WebSockets, HTTPS, and routes. Don't use `express`.
- `bun:sqlite` for SQLite. Don't use `better-sqlite3`.
- `Bun.redis` for Redis. Don't use `ioredis`.
- `Bun.sql` for Postgres. Don't use `pg` or `postgres.js`.
- `WebSocket` is built-in. Don't use `ws`.
- Prefer `Bun.file` over `node:fs`'s readFile/writeFile
- Bun.$`ls` instead of execa.

## Testing

Use `bun test` to run tests.

```ts#index.test.ts
import { test, expect } from "bun:test";

test("hello world", () => {
  expect(1).toBe(1);
});
```

## Frontend

Use HTML imports with `Bun.serve()`. Don't use `vite`. HTML imports fully support React, CSS, Tailwind.

Server:

```ts#index.ts
import index from "./index.html"

Bun.serve({
  routes: {
    "/": index,
    "/api/users/:id": {
      GET: (req) => {
        return new Response(JSON.stringify({ id: req.params.id }));
      },
    },
  },
  // optional websocket support
  websocket: {
    open: (ws) => {
      ws.send("Hello, world!");
    },
    message: (ws, message) => {
      ws.send(message);
    },
    close: (ws) => {
      // handle close
    }
  },
  development: {
    hmr: true,
    console: true,
  }
})
```

HTML files can import .tsx, .jsx or .js files directly and Bun's bundler will transpile & bundle automatically. `<link>` tags can point to stylesheets and Bun's CSS bundler will bundle.

```html#index.html
<html>
  <body>
    <h1>Hello, world!</h1>
    <script type="module" src="./frontend.tsx"></script>
  </body>
</html>
```

With the following `frontend.tsx`:

```tsx#frontend.tsx
import React from "react";

// import .css files directly and it works
import './index.css';

import { createRoot } from "react-dom/client";

const root = createRoot(document.body);

export default function Frontend() {
  return <h1>Hello, world!</h1>;
}

root.render(<Frontend />);
```

Then, run index.ts

```sh
bun --hot ./index.ts
```

For more information, read the Bun API docs in `node_modules/bun-types/docs/**.md`.

## G-code Rendering Reference (BambuStudio)

Based on analysis of BambuStudio source code, here's how professional slicers render G-code toolpaths:

### Key Finding: NO SPLINES USED
**BambuStudio does NOT use splines or curves** for rendering toolpaths. They use precise straight-line segments with proper connectivity.

### Reference Implementation Files

**1. GCodeProcessor (Arc Interpolation)**
- `BambuStudio/src/libslic3r/GCode/GCodeProcessor.cpp` (lines 4598-5050)
- `BambuStudio/src/libslic3r/GCode/GCodeProcessor.hpp` (lines 196-234)

**2. Renderers (Geometry Generation)**
- `BambuStudio/src/slic3r/GUI/GCodeRenderer/LegacyRenderer.cpp` (lines 1038-1216)
- `BambuStudio/src/slic3r/GUI/GCodeRenderer/AdvancedRenderer.cpp`
- `BambuStudio/src/slic3r/GUI/GCodeRenderer/BaseRenderer.hpp`

### Arc Interpolation Algorithm

**Tolerance-based approach (NOT fixed segment count):**
```cpp
static const float DRAW_ARC_TOLERANCE = 0.0125f;  // 12.5 microns

// Calculate maximum angular step based on tolerance
float radian_step = 2 * acos((radius - DRAW_ARC_TOLERANCE) / radius);

// Generate points using rotation matrix (NOT splines)
for (auto i = 0; i < interpolation_num; i++) {
    float cos_val = cos((i+1) * radian_step);
    float sin_val = sin((i+1) * radian_step);
    points[i] = Vec3f(
        center.x + delta.x * cos_val - delta.y * sin_val,
        center.y + delta.x * sin_val + delta.y * cos_val,
        start.z + (i + 1) * z_step
    );
}
```

### Key Techniques for Continuous Appearance

**1. Tolerance-Based Subdivision**
- Use chord height formula: `chord_height = radius * (1 - cos(θ/2))`
- Calculate angular step: `θ = 2 * acos((radius - tolerance) / radius)`
- 0.0125mm tolerance ensures smooth curves without over-segmentation

**2. Proper Segment Connectivity**
```cpp
// From LegacyRenderer.cpp:1184-1193
if (is_first_segment) {
    // Full 8-vertex cap
    append_starting_cap_triangles(indices, first_seg_v_offsets);
    append_stem_triangles(indices, first_seg_v_offsets);
    append_ending_cap_triangles(indices, first_seg_v_offsets);
} else {
    // Continuing segment: reuse previous vertices
    // Only add 6 new vertices, connect to previous 4
}
```

**3. Data Structure: MoveVertex**
```cpp
struct MoveVertex {
    EMoveType type;
    Vec3f position;
    Vec3f arc_center_position;
    std::vector<Vec3f> interpolation_points;  // For G2/G3
    float width, height;
};
```

### What NOT To Do

❌ **Catmull-Rom splines** - Over-smooths the toolpath, inaccurate
❌ **Fixed segment count** - Wrong for varying radius arcs
❌ **Bezier curves** - Don't match actual printer motion
❌ **High overlap factors** - Creates visual artifacts (spikes)

✅ **DO: Straight line segments** with tolerance-based density
✅ **DO: Proper vertex sharing** between consecutive segments
✅ **DO: Rotation matrix math** for accurate arc points
✅ **DO: 0.0125mm tolerance** for arc interpolation

### Implementation Notes

- Arcs (G2/G3) are interpolated into LINEAR segments for rendering
- Linear moves (G1) remain as-is
- The `interpolation_points` vector stores arc subdivision points
- Renderers create tube geometry with rectangular or circular cross-section
- Index buffers ensure seamless connectivity between segments
