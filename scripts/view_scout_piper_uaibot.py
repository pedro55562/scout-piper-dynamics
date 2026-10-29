"""Preview Scout Mini + Piper with prescribed v,w commands and no controller."""

import argparse
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import uaibot as ub

try:
    from .scout_piper_uaibot import MODEL_DIR, create_scout_piper
except ImportError:
    from scout_piper_uaibot import MODEL_DIR, create_scout_piper

ROOT = MODEL_DIR.parents[1]

# duration [s], forward speed [m/s], yaw rate [rad/s]. No feedback or task law.
DEMO_COMMANDS = ((2, 0, 0), (4, 0.12, 0), (4, 0.12, 0.35),
                 (3, 0, -0.4), (3, -0.10, 0), (2, 0, 0))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def create_scene(asset_url, *, static=False, dt=0.04):
    """Show the assembly; optional open-loop motion keeps all arm joints fixed."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    model = create_scout_piper(asset_url=asset_url)
    sim = ub.Simulation(
        show_world_frame=True, show_grid=False, background_color="#f1f3f6",
        camera_start_pose=[1.9, -2.3, 1.65, 0.45, 0.05, 0.4, 1.35],
        width=1200, height=800,
    )
    model.add_to_simulation(sim)
    sim.add(ub.Box(name="ground", width=5, depth=5, height=0.01,
                   htm=ub.Utils.trn([0, 0, -0.005]), color="#e3e8ed"))
    if not static:
        for duration, v, w in DEMO_COMMANDS:
            model.set_velocity(v, w)
            steps = int(np.ceil(duration / dt))
            for _ in range(steps):
                model.step(duration / steps)
    sim.add(ub.HTMLDiv(
        name="assembly_legend", html_text=(
            "<b>Scout Mini + Piper</b>"
            "<br>Montagem rígida · base móvel planar"
            "<br>" + ("Pose estática" if static else "Reta · curva · giro · marcha à ré")
            + "<br>Comandos v, ω predefinidos · sem controlador"
            "<br>Juntas do braço fixas em relação ao Scout"
            "<br>Referenciais: mundo · Scout · efetuador"
        ), style="position:absolute;left:18px;top:18px;padding:14px;background:#ffffffdd;"
                 "border-radius:8px;color:#223;font:14px sans-serif;line-height:1.6;",
    ))
    return sim, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--static", action="store_true", help="Show the assembly without motion")
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(QuietHandler, directory=str(ROOT)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        sim, _ = create_scene(f"{origin}/models/scout_piper/assets", static=args.static, dt=args.dt)
        html = '<meta charset="utf-8">\n' + sim.gen_code().replace(origin + "/models/", "../models/")
        (output / "scout_piper_uaibot.html").write_text(html, encoding="utf-8")
        url = f"{origin}/artifacts/scout_piper_uaibot.html"
        print(f"Scene: {url}", flush=True)
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
