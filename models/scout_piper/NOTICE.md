# Asset provenance

Visual OBJ files are converted from the supplied Collada meshes with scene
transforms baked and diffuse colors assigned in `scene.xml`. Collision STL files
are convex hull derivatives of supplied Piper collision meshes. Original source
paths, checksums, conversion operations, and inertial changes are recorded in
`source_manifest.json`. The source workspace is not modified.

Scout source: `ppscout_ws-dev/src/third_party/scout_ros2/`, distributed with the
Apache 2.0 license; see `licenses/Scout-Apache-2.0.txt`.

Piper source: `ppscout_ws-dev/src/third_party/agx_arm_urdf/`, distributed with the
MIT license, Copyright (c) 2026 aalicecc; see `licenses/Piper-MIT.txt`.

The combined MJCF is a simplified derivative for initial visualization and
simulation. Physical approximations and all model changes are described in
`../../docs/model_notes.md`.
