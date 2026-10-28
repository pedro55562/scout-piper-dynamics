# Scout Mini + Piper in MuJoCo

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
