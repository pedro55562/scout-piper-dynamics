#!/usr/bin/env python3
"""Run the free robot under gravity with all six torque commands held at zero."""

import argparse

import mujoco
import numpy as np
from arm_dynamics import ARM_JOINTS
from viewer import load_model


def run(seconds=5.0):
    if not np.isfinite(seconds) or seconds < 0.25:
        raise ValueError("Use a finite duration of at least 0.25 seconds")
    if mujoco.get_mjcb_control() is not None or mujoco.get_mjcb_passive() is not None:
        raise RuntimeError(
            "The uncontrolled test requires no control or passive-force callback"
        )
    model, data = load_model()
    np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])
    np.testing.assert_array_equal(model.body_gravcomp, 0)
    addresses = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
    initial = data.qpos[addresses].copy()
    max_change = np.zeros(6)
    max_torque = 0.0
    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
        # Constant zero inputs; no state feedback, gravity compensation, or applied wrenches.
        data.ctrl[:] = 0
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_step(model, data)
        max_torque = max(max_torque, float(np.max(np.abs(data.qfrc_actuator))))
        if (
            np.any(data.ctrl)
            or np.any(data.actuator_force)
            or np.any(data.qfrc_actuator)
        ):
            raise RuntimeError("An actuator produced nonzero torque")
        if not all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc)):
            raise RuntimeError("Nonfinite simulation state")
        max_change = np.maximum(max_change, abs(data.qpos[addresses] - initial))
    if any(w.number for w in data.warning):
        raise RuntimeError("MuJoCo reported a physics warning")
    if not np.isclose(data.time, steps * model.opt.timestep, atol=1e-9, rtol=0):
        raise RuntimeError("Unexpected simulation time reset")
    if max_change.max() < 0.01:
        raise RuntimeError("Arm did not move appreciably under gravity")
    print(
        f"Zero-torque gravity test: {steps} steps, {data.time:.3f} s, no physics warnings."
    )
    print(
        f"Gravity: {model.opt.gravity} m/s²; gravity compensation: zero on every body."
    )
    print(
        f"All ctrl, actuator_force and qfrc_actuator stayed zero (peak {max_torque:g} N·m)."
    )
    print("Joint    initial (rad)   final (rad)   max |change| (rad)")
    for i, name in enumerate(ARM_JOINTS):
        print(
            f"{name:8s} {initial[i]:12.6f} {data.qpos[addresses[i]]:13.6f} {max_change[i]:19.6f}"
        )
    print(
        "Contact, joint-limit and gripper constraint forces remain active; see docs/physical_model_report.md."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Duration, at least 0.25 s (default: 5)",
    )
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0.25:
        parser.error("--seconds must be finite and at least 0.25")
    run(args.seconds)
