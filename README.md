# Scout Mini + Piper in MuJoCo

## Piper in uaibot

The project-local factory in `scripts/piper_uaibot.py` builds the standard
six-axis Piper from the pinned URDF and existing OBJ meshes, without changing
uaibot. Run the circular tracking demo showing the robot, TCP, moving target,
world frame and desired circle (no DH or collision display):

```bash
source .venv/bin/activate
python scripts/view_piper_uaibot.py
```

The command opens `http://127.0.0.1:8000/artifacts/piper_uaibot.html` and keeps
the local asset server running until Ctrl+C. Use `--no-open`, `--port 8001`,
or `--q 0 1 -1.2 0 0.2 0` as needed. Tune the controller with `--gain 2`,
`--damping 1e-6`, `--duration 40`, and `--dt 0.02`.
The default circle has a **200 mm radius (400 mm diameter)** in world YZ,
a 20 second period, and runs for two turns. Its center is approximately
`[0.30, 0, 0.42]` m, starting at the top. `CIRCLE_Q0` places the arm at that
starting pose. The gripper follows the arm's lateral heading to keep the
wrist within its limits; angular feedforward follows that rotation too.
Use `--radius 0.20` and `--period 20` to change its geometry and speed.
The complete circumference is always displayed using `ub.PointCloud`.
A cyan `ub.Ball` shows the current **desired** position, animated from the
same reference samples used by the controller, independently of actual TCP error.
`--export-only` generates the HTML, `artifacts/piper_dh.json` and the trajectory
`artifacts/piper_control.npz` and exits; then serve the repository using
`python -m http.server 8000 --bind 127.0.0.1` to view the HTML. Meshes stay
local; uaibot's viewer loads its JavaScript dependencies from the internet.

```python
import uaibot as ub
from scripts.piper_uaibot import create_piper

piper = create_piper(gripper_opening=0.06, show_dh=False)
robot = piper.robot                       # native uaibot.Robot
sim = ub.Simulation(show_grid=False)
piper.add_to_simulation(sim, show_collision=False)
piper.set_configuration([0, 1, -1.2, 0, 0.2, 0])  # URDF angles, radians
frames = robot.fkm(axis="dh")
# For direct robot APIs, convert encoder/URDF angles to full DH angles:
q_dh = piper.to_uaibot_q([0, 1, -1.2, 0, 0.2, 0])
# distances = robot.compute_dist(obstacle, q=q_dh)
```

DH uses `Rz(theta) Tz(d) Tx(a) Rx(alpha)` with offsets derived from the URDF;
the original rounded RPY and small geometric offsets are retained. Uaibot
uses the full theta as `q`, rather than adding `Link.theta` for revolute joints.
The factory therefore shifts both the initial configuration and joint limits.
`piper.kinematics.dh` has columns `[theta_offset, d, alpha, a]` in rad/m;
the JSON export also records the base and TCP transforms.
The nominal end-effector is centered in the gripping region between the
fingers, 104.25 mm along flange +z. This places it 38.25 mm behind the
fingertips: 4.5 mm mount + 138 mm finger frame - half the 76.5 mm finger
length measured from the STL.
TCP +z points outward along the approach direction; +y follows the jaw
closing axis, and +x completes the right-handed frame. Override `tcp_offset`
in the factory if a calibrated TCP is available. The gripper opening is fixed at
construction (total gap, 0 to 0.1 m), and is not a seventh joint.

The controller in `scripts/piper_kinematic_control.py` directly uses uaibot:

```python
r, jac_r = robot.task_function(target)
feedforward = task_feedforward(robot, target, target_twist)
qdot = ub.Utils.dp_inv(jac_r, eps=damping) @ (-k * r + feedforward)
```

`run_kinematic_control` integrates joint velocities with Euler at 20 ms.
`CircularTrajectory(t)` supplies both the desired pose and its analytical
world twist `[v_des, omega_des]`. The same runner also accepts a fixed pose.
Since `rdot = Jr*qdot + partial_t(r)`, the feedforward cancels the explicit
target derivative. For a fixed-orientation trajectory it is `[v_des; 0]`.
For rotating targets the orientation entries are
`dot(cross(omega_des, axis_des), axis_actual)`, matching uaibot's task function.
The demo scales velocities to at most 1 rad/s and clips positions to joint
limits. Damping and these limits make `rdot = -k*r` approximate. This is a
kinematic controller without collision avoidance or contact dynamics.
The task's final three entries are uaibot's axis alignment errors
`1 - dot(axis_target, axis_current)`; they are not angles in radians.
The exported history also records a separate orientation angle in radians,
desired/actual TCP poses and the desired twist.

Collision objects use the same basic primitives as uaibot's standard robots:
cylinders for the base, motor housings and circular flanges; boxes for the
elongated housings and gripper. Dimensions are fitted around the existing
STL hulls with a 0.5 mm margin. These are conservative approximations and
may report clearance smaller than the visual mesh's true clearance.
All ten moving volumes (six links, flange, gripper and fingers)
are registered through `Link.attach_col_object`, so `robot.compute_dist`
uses them. The fixed base is available separately as `piper.base_collision`;
include its `compute_dist` result when checking whole-robot clearance.
Use `piper.add_ani_frame(time, q_urdf)` to synchronize robot, TCP, DH and
collision animations. The control scene does not add the DH/collision
objects to the simulation, and provides no toggles to display them.

Validation: `python -m unittest tests.test_piper_uaibot tests.test_piper_control -v` checks DH against
URDF in 22 configurations, moving collision frames, hull enclosure and
distance queries against an external object, TCP approach direction, task
descent and convergence to a fixed target, the analytical trajectory velocity,
feedforward sign by finite differences, and circular tracking below 1 mm.

A free-moving four-wheel Scout Mini with a six-joint Piper arm and passive
gripper. Python sends joint torques to MuJoCo and reads the exact simulated
state. There are no controllers, sensors, estimators or gravity compensation.

## Install and run

From the repository root (tested with Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/viewer.py --joints
```

The viewer starts paused. **Space** runs/pauses; **R** resets home; the mouse
rotates/pans/zooms. Use `--run`, `--collisions` or `--headless` as needed.
On macOS use `mjpython scripts/viewer.py`. Headless simulation needs no display.

For the complete development environment, including Uaibot, the asset build
tools, NumPy/SciPy, symbolic math, control systems, plotting and notebooks:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m jupyterlab
```

Select the Python kernel from `.venv` when opening notebooks in an editor.
`requirements-dev.txt` also includes pytest and Ruff for development.

## Python interface

```python
import numpy as np
from scripts.scout_piper_sim import ScoutPiperSim

sim = ScoutPiperSim()
sim.reset(
    base_q=[0.0, 0.0, 0.0],
    base_dq=[0.0, 0.0, 0.0],
    arm_q=[0.0, 0.7, -1.4, 0.0, 0.7, 0.0],
)

for _ in range(1000):
    state = sim.get_state()
    qb, dqb = state.base.q, state.base.dq
    qa, dqa = state.arm.q, state.arm.dq
    # Future controllers will compute these torques.
    sim.set_torques(wheels=np.zeros(4), arm=np.zeros(6))
    sim.step()
```

Each state group has `q`, `dq` and `ddq` arrays; `state.time` is seconds:

| Group | Shape | Configuration |
| --- | --- | --- |
| `state.base` | `(3,)` | World `[x, y, theta]`, meters/radians; derivatives in the world configuration |
| `state.arm` | `(6,)` | Piper joint angles, radians |
| `state.wheels` | `(4,)` | Wheel angles in front-left, front-right, rear-left, rear-right order |

Velocities/accelerations use the corresponding SI units. Arrays are independent
snapshots. Heading is wrapped to `[-pi, pi]`; derivatives account for base tilt.
A vertical forward axis has undefined planar heading and raises `ValueError`.

`reset` also accepts `arm_dq`, `wheel_q` and `wheel_dq`; omitted values use home
and zero velocities. It preserves nominal ground clearance and zeros commands.
The full floating base remains free during simulation. `step(n=10)` advances ten
fixed steps; `sim.timestep` gives the step duration.

All ten motors use `gear=1`, with inputs in joint-side N·m. Positive wheel torque
rolls forward. Limits are ±5 N·m per wheel (configurable software bounds, not
verified manufacturer ratings), ±32 N·m on arm joints 1–3 and ±6.506016 N·m on
joints 4–6. Override wheel bounds with
`ScoutPiperSim(wheel_torque_limits=[5, 5, 5, 5])` or change
`scout.torque_limits_Nm` in `models/scout_piper/build_config.json` before rebuilding.

Use `set_arm_torques`, `set_wheel_torques`, `set_torques(arm=..., wheels=...)`
and `zero_torques()`. Commands persist between steps and saturate at motor limits;
omitted groups stay unchanged. `sim.arm_torques` / `sim.wheel_torques` return
current saturated commands; `arm_torque_limits` / `wheel_torque_limits` return
positive bounds. Invalid shapes or nonfinite inputs raise `ValueError`.

For an API-driven GUI loop, use `with sim.launch_viewer() as viewer:` and call
`viewer.sync()` after stepping (pace the loop as needed). Manual GUI controls
can apply nonzero torques; use the dedicated test below for zero-input behavior.

## Tests and regeneration

```bash
python -m unittest discover -s tests -v
python scripts/validate_model.py
python scripts/zero_torque.py --seconds 5
```

The tests cover state/reset/frame conversions, all ten signed torque inputs,
saturation, forward/backward driving, turning with slip, combined motion and
gravity behavior. Existing model checks cover structure, inertias and geometry.

Optional regeneration retains the torque motors:

```bash
python -m pip install -r requirements-assets.txt
python scripts/build_model.py --fetch
```
