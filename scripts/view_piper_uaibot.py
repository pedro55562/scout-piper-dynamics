"""Show circular TCP tracking with feedforward, target and world frames."""

import argparse
import json
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import uaibot as ub

try:
    from .piper_kinematic_control import (
        CIRCLE_Q0,
        CircularTrajectory,
        run_kinematic_control,
    )
    from .piper_uaibot import MODEL_DIR, create_piper
except ImportError:
    from piper_kinematic_control import (
        CIRCLE_Q0,
        CircularTrajectory,
        run_kinematic_control,
    )
    from piper_uaibot import MODEL_DIR, create_piper


ROOT = MODEL_DIR.parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def create_scene(asset_url, q0=None, *, k=2.0, damping=1e-6, duration=40.0, dt=0.02,
                 radius=0.20, period=20.0):
    """Return the clean control scene, Piper model and simulation history."""
    piper = create_piper(asset_url=asset_url, q0=CIRCLE_Q0 if q0 is None else q0, show_dh=False)
    sim = ub.Simulation(
        show_grid=False, show_world_frame=True, background_color="#f1f3f6",
        camera_start_pose=[1.0, -1.1, 0.8, 0.12, 0, 0.27, 1.6],
        width=1200, height=800,
    )
    piper.add_to_simulation(sim, show_collision=False)
    trajectory = CircularTrajectory(piper.robot.fkm(), radius=radius, period=period,
                                     start_phase=np.pi / 2, follow_heading=True)
    target_frame = ub.Frame(name="target_tcp", htm=trajectory(0)[0], size=0.05,
                            axis_color=["#ef9a9a", "#a5d6a7", "#90baff"])
    sim.add(target_frame)
    # This is the *desired* point, animated from target_pose rather than eef_pose.
    reference_ball = ub.Ball(name="reference_point", htm=trajectory(0)[0],
                             radius=0.009, color="cyan", opacity=1.0)
    sim.add(reference_ball)
    samples = [trajectory(time)[0][:3, 3] for time in np.linspace(0, period, 900)]
    sim.add(ub.PointCloud(name="circle_path", points=np.array(samples).T,
                          size=0.004, color="#546e7a"))
    result = run_kinematic_control(piper, trajectory, duration=duration, dt=dt, k=k, damping=damping)
    for time, pose in zip(result.time[1:], result.target_pose[1:]):
        target_frame.add_ani_frame(float(time), pose)
        reference_ball.add_ani_frame(float(time), pose)
    sim.add(ub.HTMLDiv(
        name="legend", html_text=(
            "<b>Piper · rastreamento circular</b>"
            "<br>Efetuador: x vermelho · y verde · <b>z azul = approach</b>"
            "<br>Mundo: na base · <b>esfera cyan: referência atual</b>"
            "<br>Circunferência completa: nuvem de pontos"
            f"<br>Círculo YZ: raio {radius * 1000:g} mm · período {period:g} s"
            f"<br>q̇ = dp_inv(Jᵣ) (−{k:g}r + feedforward)"
            f"<br>Erro RMS: {np.sqrt(np.mean(result.position_error**2)) * 1000:.2f} mm"
        ), style="position:absolute;left:18px;top:18px;padding:14px;background:#ffffffdd;"
                 "border-radius:8px;color:#223;font:14px sans-serif;line-height:1.6;",
    ))
    return sim, piper, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="Serve without opening a browser")
    parser.add_argument("--export-only", action="store_true", help="Export HTML and DH JSON, then exit")
    parser.add_argument("--q", type=float, nargs=6, default=CIRCLE_Q0.tolist(), metavar="RAD")
    parser.add_argument("--gain", type=float, default=2.0, help="Positive task gain k")
    parser.add_argument("--damping", type=float, default=1e-6, help="dp_inv eps")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=0.20, help="Circle radius in meters")
    parser.add_argument("--period", type=float, default=20.0, help="Circle period in seconds")
    args = parser.parse_args()
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(QuietHandler, directory=str(ROOT)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        sim, piper, result = create_scene(
            f"{origin}/models/scout_piper/assets", q0=args.q, k=args.gain,
            damping=args.damping, duration=args.duration, dt=args.dt,
            radius=args.radius, period=args.period,
        )
        # Generation performs HTTP HEAD checks. Store relative mesh URLs after
        # those checks, so the exported HTML can be served on any local port.
        html = '<meta charset="utf-8">\n' + sim.gen_code().replace(origin + "/models/", "../models/")
        (output / "piper_uaibot.html").write_text(html, encoding="utf-8")
        (output / "piper_dh.json").write_text(json.dumps({
            "convention": "standard: Rz(theta) Tz(d) Tx(a) Rx(alpha)",
            "columns": ["theta_offset_rad", "d_m", "alpha_rad", "a_m"],
            "rows": piper.kinematics.dh.tolist(),
            "htm_base_0": piper.kinematics.base_0.tolist(),
            "htm_n_eef": piper.robot.htm_n_eef.tolist(),
            "initial_q_urdf_rad": args.q,
            "initial_q_uaibot_rad": piper.to_uaibot_q(args.q).tolist(),
        }, indent=2) + "\n", encoding="utf-8")
        np.savez(output / "piper_control.npz", time=result.time, q_urdf=result.q_urdf,
                 task=result.task, qdot=result.qdot, position_error=result.position_error,
                 orientation_error=result.orientation_error, target_pose=result.target_pose,
                 eef_pose=result.eef_pose, target_twist=result.target_twist)
        url = f"{origin}/artifacts/piper_uaibot.html"
        print(f"Scene: {url}", flush=True)
        print("DH parameters: artifacts/piper_dh.json", flush=True)
        print(f"Final TCP error: {1000 * result.position_error[-1]:.3f} mm, "
              f"{np.degrees(result.orientation_error[-1]):.3f} degrees", flush=True)
        print(f"Tracking error: RMS={1000 * np.sqrt(np.mean(result.position_error**2)):.3f} mm, "
              f"max={1000 * result.position_error.max():.3f} mm", flush=True)
        if args.export_only:
            print(f"To view: .venv/bin/python -m http.server {args.port} --bind 127.0.0.1 --directory {ROOT}")
            return
        if not args.no_open:
            webbrowser.open(url)
        print("Ctrl+C to stop the local server.", flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
