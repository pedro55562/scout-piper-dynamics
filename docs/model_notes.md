# Model changes and validation

## Source and geometry

The initial checkout contained `ppscout_ws-dev/` already extracted, with no ZIP.
It was inspected without modification. All 411 original source files were
confirmed byte-for-byte unchanged after implementation. The source workspace
is optional and is not required to load the delivered MJCF.

The conversion uses `scout_piper.urdf.xacro`,
`piper_with_gripper_scout.xacro`, and `piper_description_scout.urdf` from
`src/ppscout_description/urdf/`, plus Scout's
`src/third_party/scout_ros2/scout_description/urdf/model.xacro`.
Original paths and checksums appear in
[`source_manifest.json`](../models/scout_piper/source_manifest.json).

Collada scene transforms are baked into OBJ vertices and split normals; source
triangles, diffuse colors, and meter units are preserved. The Scout chassis
visual rotation `rpy=(1.5708,0,-1.5708)` is baked into its assets. All retained
joint origins, axes, and position limits follow the source.

The model contains the chassis, four independent wheels, Piper base, six arm
links, flange, gripper housing, and two fingers. The tower/peripheral structure,
IMU, LiDAR, cameras, and pan/tilt bodies and joints are removed. There are no
ROS interfaces, Gazebo plugins, sensors, actuators, or controllers.

## Direct mounting transformation

The rigid transform from Scout `base_link` to `piper_base_link` is:

```text
xyz = (0, 0, 0.054) m
rpy = (0, 0, 0) rad
```

The first arm axis is centered at chassis x=y=0. The Piper casing footprint is
x `[-0.058500,0.040000]`, y `[-0.040000,0.040000]` m, with bottom z=0.
The central chassis deck below it is at z `0.053998956–0.053999318` m,
measured using an 11-by-11 ray grid. A conservative bound over every chassis
triangle whose projected bounding box intersects that footprint is
z `0.053999688` m. This verifies clearance across the entire footprint.
The submicrometer clearance reflects source export rounding: the nominal
mount is flush, with no support structure, plate, or added mount mass.

The chassis's higher side rails reach about z=0.069 m but lie outside this
footprint. The original z=0.19 m offset would leave approximately 0.136 m
between the Piper base and central deck. Structural fastening is not modeled.

## Physical parameters

All retained physical-link masses come from the source, totaling **76.7 kg**:
60 kg chassis, four 3 kg wheels, and 4.7 kg arm/gripper assembly. These source
values have not been replaced with assumed Scout Mini datasheet values.

Source tensors are interpreted about their declared COM and rotated into the
link frame. In addition to positive principal moments and triangle inequalities,
the audit checks COM bounds and the necessary bounded-variance condition:

```text
S = (trace(I)/2 * identity - I) / mass
S_ii <= (COM_i - mesh_min_i) * (mesh_max_i - COM_i)
```

For example, link2's source x variance is 0.057790 m² versus a possible maximum
of 0.024495 m² within its mesh at the declared COM. The wheel source z variance
is 0.216350 m² versus a maximum of 0.003995 m². Such tuples cannot represent
mass contained within their supplied geometry at the stated COM.

Original COM/tensors are preserved for **link1, link4, and the flange**. For
the chassis, wheels, Piper base, link2, link3, link5, link6, gripper housing, and
fingers, rejected COM/tensor tuples are replaced by uniform-density convex-hull
integration at the original mass. Piper uses the supplied collision STL;
Scout uses transformed visual geometry. These are documented approximations,
not measured hardware data. No hidden mass or inertia clamping is used.
The manifest records every original/final tuple, bound, and rejection reason.

The source's geometry-free opening-coordinate `gripper_link` and its artificial
0.001 kg mass are removed. Both physical fingers retain their frames and ranges:

```text
q1 =  opening/2   (0 to 0.05 m)
q2 = -opening/2   (-0.05 to 0 m)
```

One joint equality enforces q2=-q1. The removed opening slider's damping
0.5 N·s/m and friction 0.05 N are distributed equally while preserving
dissipated power: each finger has damping 1 N·s/m and friction 0.05 N.
Other retained passive joint parameters follow the source. Source effort and
velocity limits are not active motor limits because there are no actuators.

## Collision and numerical assumptions

- The base has a free joint. There are 16 robot bodies, 13 joints (one free,
  ten hinge, two slide), nq=19, nv=18, and one gripper equality.
- Ground is an infinite plane at z=0; gravity is `(0,0,-9.81)` m/s².
- The chassis collision shape is a box spanning its visual x/y bounds and
  underside, with top at the central deck. Suspension is not articulated.
- Each tire has a cylindrical collision shape, excluding decorative axle
  extensions: radius 0.080023891 m, half-width 0.041856307 m, wheel-frame center
  `(-0.000007551,-0.001018431,0.039638345)` m.
- Piper collision shapes are convex hulls of supplied collision meshes.
  These close cavities and limit detailed grasp/contact fidelity.
- Visuals are contact-disabled in group 2. Collision shapes are in group 3,
  hidden by the viewer unless `--collisions` is supplied. Standard welded-body
  and parent contact filtering is retained; other self collisions are enabled.
- Friction `(1,0.005,0.0001)` uses MuJoCo defaults and is uncalibrated.
  Contact compliance otherwise uses defaults.
- Numerical choices are `implicitfast`, 0.001 s timestep, elliptic friction
  cones, and gripper equality `solref=(0.002,1)`,
  `solimp=(0.99,0.999,0.001)`. Contacts and equality are soft constraints.

The chassis starts at world z=0.182040621 m, derived from the lowest rotated
tire envelope plus 1 mm clearance. Source left/right wheel offsets differ by
1 mm, so the chassis settles with a slight roll. The home arm angles are
`(0,0.7,-1.4,0,0.7,0)` rad with 0.04 m gripper opening. This is an inspection
pose: the unpowered arm falls and the mobile base can roll when stepped.

## Validation and remaining limitations

Initial validation used Python 3.12, MuJoCo 3.17.0, and NumPy 2.5.4. Seven
checks passed, covering exact body/joint inventories, standalone assets,
mounting clearance on compiled geometry, inertial consistency, source FK for
all retained bodies in 20 seeded configurations, gripper openings from 0 to
0.1 m, and passive simulation.

The 10,000-step / 10-second run had finite state, zero physics warnings, and
positive final ground reaction forces at all four wheels. Maximum settling
penetration was about 3.55 mm; maximum finger symmetry residual was 0.0421 mm.
A standalone copy containing only the model/assets and viewer also ran from
`/tmp` without the original workspace. EGL rendering succeeded and was
visually inspected.

The desktop viewer opened and synchronized frames, but this machine reports
`OpenGL error 0x502 ... mjr_makeContext` even for a minimal sphere/plane GUI
scene. Waiting for its render thread to finish resolved an initial shutdown
race. Headless physics and EGL rendering did not show that warning.

Run `python scripts/validate_model.py` to repeat validation. Source FK comparison
is skipped when the optional source workspace is absent. Hardware masses,
internal mass distributions, tire friction, and detailed collision geometry
need confirmation before quantitative dynamics or grasp studies. This task
does not include analytical dynamics, UAIBot integration, or custom controllers.
