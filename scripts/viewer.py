#!/usr/bin/env python3
"""Inspect the standalone robot, or step it without a display. No controllers."""

import argparse
import math
from pathlib import Path
import threading
import time

import mujoco


MODEL_PATH = Path(__file__).resolve().parents[1] / "models/scout_piper/scene.xml"


def load_model(path=MODEL_PATH):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Step and exit without opening a window")
    parser.add_argument("--steps", type=int, default=5000, help="Number of headless steps (default: 5000 = 5 s)")
    parser.add_argument("--run", action="store_true", help="Start the viewer running; default is paused for inspection")
    parser.add_argument("--joints", action="store_true", help="Show joint axes")
    parser.add_argument("--collisions", action="store_true", help="Show collision geometry (group 3)")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    model, data = load_model()
    print(f"Loaded {model.nbody - 1} robot bodies, {model.njnt} joints, "
          f"nq={model.nq}, nv={model.nv}, {model.neq} gripper equality.")
    print("Joints: " + ", ".join(model.joint(i).name for i in range(model.njnt)))
    if args.headless:
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        if not all(math.isfinite(float(x)) for x in [*data.qpos, *data.qvel, *data.qacc]):
            raise RuntimeError("Nonfinite simulation state")
        warnings = sum(int(w.number) for w in data.warning)
        if warnings or abs(data.time - args.steps * model.opt.timestep) > 1e-8:
            raise RuntimeError(f"Simulation warning or unexpected time reset: {warnings} warnings, t={data.time}")
        print(f"Stepped {args.steps} times: t={data.time:.3f} s, contacts={data.ncon}, "
              f"base z={data.qpos[2]:.4f} m, warnings={warnings}.")
        return

    from mujoco import viewer as mj_viewer

    running = threading.Event()
    reset = threading.Event()
    if args.run:
        running.set()

    def key_callback(key):
        if key == 32:  # Space
            if running.is_set():
                running.clear()
            else:
                running.set()
        elif key == ord("R"):
            reset.set()

    print("Space: pause/run. R: reset home. Mouse: rotate/pan/zoom. "
          "Expand Joint in the right panel to inspect joint positions while paused.")
    existing_threads = set(threading.enumerate())
    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.43]
        viewer.cam.distance = 1.65
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.opt.geomgroup[3] = int(args.collisions)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = args.joints
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                if reset.is_set():
                    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                    reset.clear()
                if running.is_set():
                    mujoco.mj_step(model, data)
                else:
                    # Updates transforms when joint sliders change in the paused viewer.
                    mujoco.mj_forward(model, data)
            viewer.sync()
            interval = model.opt.timestep if running.is_set() else 1 / 60
            time.sleep(max(0, interval - (time.monotonic() - start)))
    # MuJoCo launches a daemon rendering thread. Let it release GLFW before Python exits.
    for thread in set(threading.enumerate()) - existing_threads:
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
