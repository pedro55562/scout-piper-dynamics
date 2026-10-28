# Asset provenance

Visual OBJ files are converted from the supplied Collada meshes with scene
transforms baked and diffuse colors assigned in `scene.xml`. Collision STL files
are convex hull derivatives of official Piper collision meshes. Source commits,
URLs, checksums, operations, parameter classifications, and inertial changes are
recorded in `upstream_lock.json` and `source_manifest.json`.

Scout Mini geometry/description: AgileX `scout_ros`, commit
`01e07881cdc566c3a657e288c59a75577992d13e`. Its description package declares BSD;
the upstream BSD notice is preserved in `licenses/Scout-BSD.txt`.

Standard Piper and gripper: AgileX `agx_arm_urdf`, commit
`680fa925cd00db7e2628563d2edc5540963045de`, MIT License,
Copyright (c) 2026 aalicecc; see `licenses/Piper-MIT.txt`.

The combined MJCF is a simplified derivative for initial visualization and
simulation. Physical approximations and all model changes are described in
`../../docs/physical_model_report.md`. The old Gazebo archive/workspace is not an
input to this rebuild. Public manuals are linked and hashed, not redistributed.
