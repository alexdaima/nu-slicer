# Mesh Module

## Purpose

The `mesh` module handles triangle mesh loading, processing, and manipulation. It provides the entry point for 3D model data into the slicing pipeline, converting STL files (and eventually other formats) into the internal `TriangleMesh` representation.

## What This Module Does

1. **STL Loading** (`stl.rs`): Parse binary and ASCII STL files into triangle meshes
2. **Mesh Representation** (`triangle_mesh.rs`): Core `TriangleMesh` and `Triangle` types
3. **Mesh Queries**: Bounding box, volume, surface area calculations
4. **Mesh Validation**: Check for common mesh issues (holes, non-manifold edges)

## libslic3r Reference

| Rust File | C++ File(s) | Description |
|-----------|-------------|-------------|
| `mod.rs` | - | Module exports |
| `stl.rs` | `Format/STL.cpp` | STL file parsing |
| `triangle_mesh.rs` | `TriangleMesh.hpp/cpp` | Mesh data structure |

### Key C++ Classes

- **`TriangleMesh`** (`TriangleMesh.hpp`): Main mesh container
- **`stl_file`** (admesh library): Low-level STL handling
- **`indexed_triangle_set`**: Vertex-indexed triangle storage

## Key Types

### Triangle

A single triangle face defined by three 3D points.

```rust
pub struct Triangle {
    pub vertices: [Point3F; 3],  // Three vertices in mm
    pub normal: Point3F,         // Face normal vector
}
```

### TriangleMesh

Collection of triangles forming a 3D solid.

```rust
pub struct TriangleMesh {
    triangles: Vec<Triangle>,
    bounding_box: BoundingBox3F,
    // Cached properties...
}

// Key methods:
mesh.from_stl(path)       // Load from STL file
mesh.triangles()          // Get triangle iterator
mesh.bounding_box()       // Get AABB
mesh.volume()             // Calculate volume (signed)
mesh.surface_area()       // Calculate surface area
mesh.is_valid()           // Check mesh integrity
mesh.repair()             // Fix common issues
```

## STL File Format

### Binary STL

```
Header (80 bytes)         - Usually ignored
Triangle count (4 bytes)  - uint32 little-endian
For each triangle:
  Normal (12 bytes)       - 3x float32
  Vertex 1 (12 bytes)     - 3x float32
  Vertex 2 (12 bytes)     - 3x float32
  Vertex 3 (12 bytes)     - 3x float32
  Attribute (2 bytes)     - Usually 0
```

### ASCII STL

```
solid name
  facet normal ni nj nk
    outer loop
      vertex v1x v1y v1z
      vertex v2x v2y v2z
      vertex v3x v3y v3z
    endloop
  endfacet
  ...
endsolid name
```

## File Structure

```
slicer/src/mesh/
├── AGENTS.md          # This file
├── mod.rs             # Module exports
├── stl.rs             # STL file loading/saving
└── triangle_mesh.rs   # TriangleMesh type (if separate)
```

## Mesh Processing Pipeline

```
STL File
    │
    ▼
┌─────────────────┐
│   load_stl()    │  Parse file, create triangles
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ TriangleMesh    │  In-memory representation
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   validate()    │  Check for issues
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    repair()     │  Fix holes, normals (if needed)
└─────────────────┘
    │
    ▼
  To slice/ module
```

## Common Mesh Issues

| Issue | Description | Fix |
|-------|-------------|-----|
| Holes | Missing triangles | Fill holes algorithm |
| Non-manifold | Edge shared by >2 faces | Remove/split |
| Inverted normals | Inconsistent winding | Recalculate normals |
| Degenerate triangles | Zero-area faces | Remove |
| Duplicate vertices | Same position, different indices | Merge |

## Volume Calculation

Volume of a mesh using the signed tetrahedron method:

```rust
fn volume(&self) -> f64 {
    let mut volume = 0.0;
    for tri in &self.triangles {
        // Signed volume of tetrahedron from origin to triangle
        let v1 = tri.vertices[0];
        let v2 = tri.vertices[1];
        let v3 = tri.vertices[2];
        volume += v1.dot(v2.cross(v3)) / 6.0;
    }
    volume.abs()
}
```

## Surface Area Calculation

```rust
fn surface_area(&self) -> f64 {
    self.triangles.iter()
        .map(|tri| tri.area())
        .sum()
}

impl Triangle {
    fn area(&self) -> f64 {
        let v1 = self.vertices[1] - self.vertices[0];
        let v2 = self.vertices[2] - self.vertices[0];
        v1.cross(v2).length() / 2.0
    }
}
```

## Future Enhancements

### Additional File Formats

| Format | Status | Priority |
|--------|--------|----------|
| STL (binary) | ✅ Implemented | - |
| STL (ASCII) | ✅ Implemented | - |
| 3MF | 📋 Planned | High |
| OBJ | 📋 Planned | Medium |
| AMF | 📋 Planned | Low |

### Mesh Operations (from libslic3r)

- `merge()` - Combine multiple meshes
- `split()` - Separate disconnected components
- `transform()` - Apply transformation matrix
- `cut()` - Cut mesh with plane
- `simplify()` - Reduce triangle count

## Dependencies

- `crate::geometry::{Point3F, BoundingBox3F}` - 3D point and bbox types
- `std::io` - File I/O
- `std::path::Path` - File paths

## Related Modules

- `slice/` - Consumes TriangleMesh, produces layer slices
- `geometry/` - Provides Point3F, BoundingBox3F types

## Testing Strategy

1. **Unit tests**: Triangle area, volume calculations
2. **File tests**: Load known STL files, verify triangle count
3. **Round-trip tests**: Load → Save → Load, compare results
4. **Error handling**: Invalid files, corrupt data
5. **Reference tests**: Compare bounding box, volume with BambuStudio