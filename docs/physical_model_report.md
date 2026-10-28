# Scout Mini + standard Piper: physical model reconstruction

Research date: **2026-10-28**. This is a physically consistent, reproducible
approximation with identified uncertainties. Public data do not establish a
fully calibrated hardware dynamics model.

## Official sources and model selection

| Repository | Exact revision | Selection/findings |
| --- | --- | --- |
| [scout_ros2](https://github.com/agilexrobotics/scout_ros2/tree/bf0110ff44cc70cdda2931d69f0e2669a3e26b9d) | `bf0110ff44cc70cdda2931d69f0e2669a3e26b9d`, humble | Investigated first. Checked description tree supplies Scout v2 geometry, despite Mini driver support. Main also lacks a dedicated Mini description. Not selected as Mini geometry. |
| [scout_ros](https://github.com/agilexrobotics/scout_ros/tree/01e07881cdc566c3a657e288c59a75577992d13e) | `01e07881cdc566c3a657e288c59a75577992d13e`, master | Selected `scout_mini.urdf.xacro` and wheel macro/meshes. Mini description last changed at `29cd883b707197e4218d899800225273afb07038` (2023). Its mass/inertia values are audited, not blindly trusted. |
| [agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf/tree/680fa925cd00db7e2628563d2edc5540963045de) | `680fa925cd00db7e2628563d2edc5540963045de`, main | Selected standard `piper/`, not `piper_x`, `piper_h` or `piper_l`, and shared `agx_gripper/`. Includes October CAD-based link6 repair and repaired gripper inertias. |
| [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm/tree/d90e234ba8e408cfa964b38aa9120b3131fc9f85) | `d90e234ba8e408cfa964b38aa9120b3131fc9f85`, master | Standard-Piper joint limits cross-check and explicitly selected DEFAULT firmware torque-command envelope. |
| [UGV_GAZEBO_SIM](https://github.com/agilexrobotics/ugv_gazebo_sim/tree/27633a956c845903ee630538afeb17fe70afdd84) | `27633a956c845903ee630538afeb17fe70afdd84` | Cross-check for Mini tire cylinder dimensions and analytical wheel inertias. Alternative Mini files contain conflicting masses and extra attachments. |
| [piper_ros](https://github.com/agilexrobotics/piper_ros/tree/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc) | `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`, noetic | Older alternatives have different frames/mass distributions. No-gripper URDF totals 4.17 kg; v100 model with fingers only 1.3621 kg. Not treated as interchangeable. |
| [agx_arm_sim](https://github.com/agilexrobotics/agx_arm_sim/tree/f8cd8b147c75d59e14f90fb0646770eefa268ed0) | `f8cd8b147c75d59e14f90fb0646770eefa268ed0` | Official MuJoCo alternative totals about 2.3669 kg, with different frames, position servos and gravity compensation. `piper_body.xml` additionally uses damping 2000/finger damping 1000. Not imported. |

The [Scout Mini manual](https://cdn.shopify.com/s/files/1/0551/0630/6141/files/SCOUT_MINI_USER_MANUAL_AgileX.pdf?v=1792636734)
identifies **23 kg kerb mass**, 451 mm wheelbase, 490 mm tire track, four
250 W independent drives, 4.3:1 reduction, and independent rocker-arm suspension.
This is the off-road differential Mini, not Scout 2.0 or the mecanum variant.
The cover says V2.0.1/2023.08, while its revision table lists V2.0.4/2024-06-19;
both identifiers and the downloaded PDF hash are recorded.

The [standard Piper manual, V1.0/2025.06](https://cdn.shopify.com/s/files/1/0551/0630/6141/files/PIPER_Product_User_Manual.pdf?v=1791860528)
lists a **4.2 kg arm**, an independently listed **optional 500 g gripper**,
626.75 mm reach and 70 mm base hole spacing. The
[current product page](https://global.agilex.ai/products/piper) distinguishes
standard Piper from Piper-X; no Piper-X/H/L properties are substituted.

No publicly retrievable native STEP/SolidWorks mass-property assembly was
obtained from the checked official repositories/catalog. The current
[document center](https://global.agilex.ai/pages/document-center) exposed
manuals/brochures but no usable native CAD download for these products. This
does not establish that manufacturer CAD does not exist. No contact form was
submitted. Manufacturer CAD-derived values published in commits were evaluated;
rendering meshes were not treated as internally complete material-resolved CAD.

The old extracted Gazebo workspace had no known pinned upstream revision.
Its historical input hashes and old masses are preserved in `legacy_workspace`
in the lock/manifest. An exact old commit cannot be reconstructed honestly.
No original archive/workspace is an input to the new pipeline.

## Evidence classes and parameter ledger

1. **Manufacturer-specified or explicitly CAD-derived.** Manual totals and the
   CAD-derived link6 mass. Official SDK command ranges are class-1 software
   information, not independent motor ratings.
2. **Official URDF, not independently validated.** Most individual masses and
   retained COM/tensors, even when elementary consistency checks pass.
3. **Geometry estimate.** Uniform-volume hull/cylinder integration, measured
   offsets and mounting surface; not manufacturer mass-property CAD.
4. **Assumed or conditional.** Fixed-Scout mass allocation, ideal joint losses,
   locked suspension and uncalibrated tire/contact properties.

[The manifest](../models/scout_piper/source_manifest.json) gives separate mass,
COM and inertia classes per body, original/final values and rejection reasons.
[The CSV](../models/scout_piper/inertial_parameters.csv) gives every final COM
and all six independent entries of each full symmetric 3×3 tensor in SI units.
Off-diagonal entries are tensor entries, not principal moments or negated CAD
products of inertia. Source poses/axes/ranges are class 2, with normalizations
explicitly distinguished; other choices are in `build_config.json`.

| Body | Mass kg | Mass / COM / tensor classes | Treatment |
| --- | ---: | --- | --- |
| Scout fixed assembly | 11 | 4 / 3 / 3 | Conditional residual, uniform chassis hull |
| Each wheel (four) | 3 | 2 / 3 / 3 | Official unverified mass, centered solid cylinder |
| Piper base | 1.02 | 2 / 3 / 3 | Source mass, geometry estimate for inconsistent tuple |
| link1 | 0.71 | 2 / 2 / 2 | Retain official COM/tensor |
| link2 | 1.16 | 2 / 3 / 3 | Source mass, uniform collision hull |
| link3 | 0.5 | 2 / 3 / 3 | Source mass, uniform collision hull |
| link4 | 0.38 | 2 / 2 / 2 | Retain official COM/tensor |
| link5 | 0.383 | 2 / 3 / 3 | Source mass, uniform collision hull |
| link6 | 0.01655408133 | 1 / 3 / 3 | New CAD-derived mass; current mesh conflicts with CAD tensor |
| Flange | 0.04 | 2 / 2 / 2 | Retain official COM/tensor |
| Gripper housing | 0.45 | 2 / 2 / 2 | Retain repaired official tensor |
| Each finger | 0.025 | 2 / 2 / 2 | Retain repaired official tensor |

### Mass budget without global rescaling

The 23 kg Scout kerb total includes wheels, battery and fixed running gear.
It is not a bare chassis to which another 12 kg of wheels should be added.
No trustworthy measured individual wheel mass was found. Retaining the official
macro's unverified 3 kg wheel allocation leaves `23 - 4*3 = 11 kg` for the fixed
assembly, including battery and nonrotating drivetrain/suspension parts.
This is conditional accounting, **not a manufacturer-specified 11 kg chassis**
or independent confirmation of the wheel masses. Treating that entire assigned
wheel mass as rotating is also an idealization.

Changing wheel masses must change the residual and corresponding inertia
estimates; yaw/roll and spin dynamics depend on that split. The aggregate mass
has stronger support than its internal distribution. No battery or motor masses
are claimed to have been identified.

The new arm masses plus flange total 4.20955408133 kg, consistent with a rounded
4.2 kg specification without forcing equality. Housing and fingers add exactly
0.5 kg. There is no payload, teaching device, tower, mount plate or synthetic
opening-coordinate mass:

```text
Scout complete base       23.00000000000 kg
Piper including flange     4.20955408133 kg
Piper gripper              0.50000000000 kg
Combined                 27.70955408133 kg
```

### COM and tensor reconstruction

URDF tensors are interpreted about their declared COM and rotated into link
frames. The audit checks COM bounds, positive principal moments, triangle
inequalities and the necessary bounded-variance condition:

```text
S = (trace(I)/2 * identity - I) / mass
S_ii <= (COM_i - mesh_min_i) * (mesh_max_i - COM_i)
```

The old official Mini wheel z variance is about 0.21635 m², incompatible with
an 80 mm-radius wheel. Piper link2's source x variance is 0.05779 m² versus a
maximum about 0.02450 m². Passing MuJoCo compilation alone is insufficient.

The [October 16 CAD-based link6 repair](https://github.com/agilexrobotics/agx_arm_urdf/commit/680fa925cd00db7e2628563d2edc5540963045de)
is better evidence than the old 7 g link with huge inertias. However, its z
variance is 5.9371e-6 m² versus 1.8982e-6 m² permitted by the currently published
4 mm-thick mesh at its declared COM. Underlying CAD may contain absent geometry.
The mass is retained; the original tuple is recorded and COM/inertia are
estimated from delivered geometry. We do not silently enlarge the mesh or
assume another inertia reference point.

Rejected Piper tuples use the watertight convex hull of the official collision
STL, integrated at uniform density with original mass. The fixed Scout uses its
transformed official visual hull at the conditional 11 kg. COM is the integrated
centroid, and the full tensor is evaluated about that centroid. No inertia
floors, blanket balancing or common mass scale factor are used. Hulls fill
cavities and do not identify actual battery/motor/material distributions.

Wheels use a solid cylinder, radius 0.08 m and width 0.084 m:
`I_axial=m*r²/2=0.0096`,
`I_transverse=m*(3*r²+w²)/12=0.006564 kg·m²` at 3 kg. These agree with the
analytical values in the official alternative simulator. The older macro's
`wheel_radius=0.16` conflicts with its approximately 0.16 m-diameter mesh and
is not used as a radius.

## Geometry, frames and mounting

Original Mini axle origins are x=±0.2319755, y=±0.2082515, z=-0.100998 m.
Rounded source ±1.57 wheel rotations become exact ±π/2; left coordinate signs
flip so positive rotation rolls forward on every wheel. All nominal axle
directions are +y. The wheel mesh has a radial
export offset: translating its entire assembly by
`(0.000007551,0.001018431,0)` m centers the measured tire bounds on the axle.
Cylinder COM/contact centers become `(0,0,0.039638345)` m. This class-3 correction
prevents an export offset from creating an eccentric rotating mass.

Source wheelbase is **463.951 mm**, versus manual **451 mm**. Original tire-center
track is about **495.780 mm**, versus manual **490 mm**. The original axle-joint
track is only 416.503 mm because meshes extend outward. These quantities are
not interchangeable. The manufacturer drawing takes precedence: final axle
origins are x=±0.2255, y=±0.205361655, z=-0.100998 m, giving exactly 451 mm
wheelbase and 490 mm tire-center track. Each wheel moves inward 6.4755 mm in x
and 2.8898 mm in y; mesh vertices are not globally stretched. The x positions
are class 1, y positions combine class-1 track with class-3 measured mesh offset,
and z retains the class-2 source stance. Fixed suspension visuals remain approximate.

The resulting visual envelope is approximately **611×581×250 mm**, compared
with manual 612×580×245 mm. The remaining height discrepancy follows the older
source vertical stance/mesh; it is disclosed rather than corrected with an
unsupported suspension preload. The manual's 115 mm clearance is not asserted
as a measurement of the simplified chassis box or low suspension/motor protrusions.

Standard-Piper source origins/orientations/axes are retained. Joint ranges are
J1 ±150°, J2 0–180°, J3 -170–0°, J4 ±100°, J5 ±70°, J6 ±170°. They agree with
pinned SDK presets within rounding. The old manual differs, including J6 ±120°;
the [October J6 correction](https://github.com/agilexrobotics/agx_arm_urdf/commit/a68b8e876a2c6fcb0ecee7a7943129e4d48dd864)
explicitly changes ±120° to ±170°. These selected limits are not certified
physical stops for an unspecified hardware revision. Velocity fields remain
metadata, not active limits or velocity servos.

At zero arm angles, J6 lies at about `(56.135,0,213.178)` mm, versus manual
`(56.128,0,213.266)` mm. The 0.088 mm difference is consistent with the small
URDF J6 offset; orientation agrees with 85° about y. Manual comparison allows
0.1 mm; exact URDF/MJCF FK comparison uses tight numerical tolerances.

The rigid mount from Scout `base_link` to `piper_base_link` is:

```text
xyz = (0,0,0.05401) m
rpy = (0,0,0) rad
```

The source chassis rotation `(1.5707,0,-1.5707)` causes about 9.5 µm deck-height
variation across the Piper footprint. An 11×11 ray grid and conservative bound
on all intersecting triangles verify clearance. Conservative deck top is
0.054008022 m; rounding up to 10 µm gives the mount above. Minimum grid clearance
is 5.19 µm. This is nominally flush, with no modeled spacer/plate. The arm axis
is centered at x=y=0; side rails lie outside its footprint.

## Torque actuation, passive mechanics and contact assumptions

The six named `motor` ports use joint transmission `gear=1`, fixed unit gain,
zero bias and no activation dynamics. Commands are signed joint-side N·m;
this interface convention is **not a physical 1:1 gearbox claim**.

The [pinned SDK firmware reference](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/piper/firmware_reference.md)
gives `PiperFW.DEFAULT` command magnitude `8*b*c`. The generator parses standard
Piper constants to obtain ±32 N·m on J1–3 and ±6.506016 N·m on J4–6. These
replace generic ±100 URDF effort placeholders. Other firmware profiles differ;
DEFAULT is an explicit baseline until the installed unit is known. Encoding
limits are not continuous/peak mechanical ratings. Coefficients b/c are not
substituted as physical reducers. No trustworthy physical Piper reductions,
rotor inertia, identified friction/damping or full torque-speed curves were
established for this specific standard model.

| Mechanism | Current implementation | Evidence/status |
| --- | --- | --- |
| Gravity | `(0,0,-9.81)` m/s²; all `gravcomp=0` | Enabled, no bias-force cancellation |
| Every joint | damping=0, frictionloss=0, stiffness=0, added armature=0 | Class-4 idealization where identified losses/rotor data are unavailable |
| Wheels | Four independent passive hinges | No wheel actuators, servo brakes or unknown backdrive behavior modeled |
| Suspension | Rigid at nominal stance | Real rocker arms exist; no invented spring/damper curves |
| Gripper | Source sliders plus soft q2=-q1 equality | 0–0.1 m opening; geometry-free source opening joint removed |
| Contacts | Friction `(1,0.005,0.0001)` | Class-4 uncalibrated MuJoCo defaults, no tire deformation |
| Collisions | Chassis box, round tires, one hull per Piper link | Class-3 approximation; hulls fill cavities |
| Stops/coupling | Soft joint limits, contacts and gripper constraint | Numerical reactions, not pose control |

Latest official descriptions supply no joint damping/friction. The previous
Gazebo additions (arm damping 1–2 N·m·s/rad, friction 0.1 N·m and finger losses)
are absent. There are no springs, gravity compensation, force callbacks,
tendons, applied `qfrc_bias` cancellation or fluid viscosity/density forces.
Real brakes/gearing can behave differently; this is not a hardware backdrive test.

Numerical settings: 1 ms `implicitfast`, elliptic friction cones, default contact
compliance and gripper equality `solref=(0.002,1)`, `solimp=(0.99,0.999,0.001)`.
No damping/friction/compliance tuning hides instability. Standard welded/parent
filtering remains; other self contacts are enabled. Group-2 visuals have no
contacts; group-3 collision geometry is hidden unless requested.

## Validation and reproducibility

[Validation results](validation_results.json) record scene/manifest hashes and
Python/MuJoCo/NumPy versions. Thirteen checks need no network, ROS or original
workspace, using delivered official URDF snapshots and normalized Mini frames:

- Exact 16-body/13-joint inventory, free base, six torque inputs, no sensors/tower.
- Asset/reference hashes and standalone paths; cached inputs checked when present.
- Full-footprint mount clearance from compiled meshes.
- Mass accounting, COM support, full tensors, positive principal moments and triangle inequalities.
- Every retained body in 20 seeded configurations, including arbitrary base orientations;
  independent manual zero pose, wheel placement/direction/centering and source provenance.
- Signed isolated 1:1 torque transmission, saturation at each bound, no servo bias/dynamics.
- Zero passive joint torque at nonzero velocity and zero gravity compensation.
- Gripper openings 0, 20, 50 and 100 mm.
- Zero-input gravity motion versus a stationary gravity-disabled diagnostic reference;
  the delivered model always retains gravity.
- 10,000 steps/10 seconds: finite state, no time resets or physics warnings,
  positive final wheel-ground reactions at all four wheels and exactly zero actuator torque.

Peak ground penetration is **0.942 mm**, peak overall contact overlap **5.919 mm**,
and peak finger symmetry error **0.0021 mm**. The undamped arm falls and impacts
existing stops/contacts. Remaining overlap/convex approximations limit precision
impact and grasp studies; it is not suppressed with artificial damping.
The rendered assembly was inspected. Home is `(0,0.7,-1.4,0,0.7,0)` rad with
40 mm gripper opening. GUI controls allow manual torques;
`scripts/zero_torque.py` guarantees explicit zero inputs.

A second offline rebuild reproduced all delivered model files byte-for-byte.
A missing official input was fetched into a temporary cache and verified against
its pinned hash. An isolated copy without the upstream cache passed all thirteen
checks from `/tmp`; bare MJCF initialization also ran five seconds without
physics warnings. The CSV was checked against every manifest COM/tensor entry.

`upstream_lock.json` pins commits, individual source hashes, manual PDF hashes,
investigated alternatives and legacy input hashes. The generator fetches missing
inputs, then builds offline. It emits MJCF, colored visual meshes, collision
hulls, source snapshots, manifest and CSV. Asset-tool versions are pinned;
runtime stays MuJoCo/NumPy. No ROS/Xacro expansion is needed at runtime.

## Old/new differences and remaining limits

The old 72 kg base / 76.7 kg robot is replaced with the scoped mass budget, not
a global rescaling. Geometry is pinned to official sources, six original
COM/tensor tuples are retained instead of three, repaired gripper values are
used, link6 mass and J6 range follow current standard-Piper sources, wheel
centering/frame rounding and published wheelbase/track are corrected, and torque bounds name an SDK profile.
Tower and sensors remain absent.

Internal mass distribution is still the largest uncertainty. Wheel masses,
fixed/rotating split and estimated link distributions need measurement or
manufacturer mass-property CAD. Suspension compliance, tire properties, brakes,
rotor/reducer inertia, losses and thermal torque-speed ratings remain unknown.
The older Mini mesh/stance differs from the current drawing despite the corrected
in-plane wheel layout. Use this model with those
assumptions visible, not as independent hardware identification or precision
contact ground truth. No analytical dynamics or controller is implemented.
