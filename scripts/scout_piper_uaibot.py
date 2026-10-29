"""Scout Mini and mounted Piper in uaibot, with open-loop planar kinematics.

No controller is applied. v and w are prescribed chassis velocities; wheel
angles use an ideal differential approximation for visualization. Tire slip,
suspension, contact response and coupled dynamics are not simulated here.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import uaibot as ub
from scipy.spatial.transform import Rotation

try:
    from .piper_uaibot import MODEL_DIR, PiperModel, create_piper
except ImportError:
    from piper_uaibot import MODEL_DIR, PiperModel, create_piper

WHEEL_NAMES = ("front_left_wheel_link", "front_right_wheel_link",
               "rear_left_wheel_link", "rear_right_wheel_link")


def _vector(value, size, name):
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result.copy()


def _pose(element):
    """MJCF local pose, including scalar-first quaternions."""
    htm = np.eye(4)
    htm[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    quat = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    htm[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    return htm


def _visuals(body, asset_url):
    models = []
    for geom in body.findall("geom[@class='visual']"):
        rgba = np.fromstring(geom.get("rgba"), sep=" ")
        color = "#" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])
        models.append(ub.Model3D(
            f"{asset_url.rstrip('/')}/visual/{geom.get('mesh')}.obj",
            htm=_pose(geom), mesh_material=ub.MeshMaterial(
                color=color, roughness=0.65, metalness=0.15),
        ))
    return models


@dataclass
class ScoutMiniModel:
    chassis: ub.RigidObject
    wheels: list
    wheel_origins: tuple
    wheel_axes: np.ndarray
    collision_objects: list
    collision_poses: tuple
    base_frame: ub.Frame
    nominal_height: float
    wheel_radius: float
    track_width: float
    _base_q: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _wheel_q: np.ndarray = field(default_factory=lambda: np.zeros(4), repr=False)
    _command: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)
    time: float = 0.0

    @property
    def base_q(self):
        """World [x, y, yaw] in meters/radians; yaw is continuous."""
        return self._base_q.copy()

    @property
    def wheel_q(self):
        """URDF wheel angles in front-left/right, rear-left/right order."""
        return self._wheel_q.copy()

    @property
    def command(self):
        """Prescribed [v, w] in m/s and rad/s."""
        return self._command.copy()

    @property
    def htm(self):
        x, y, yaw = self._base_q
        htm = np.eye(4)
        htm[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        htm[:3, 3] = [x, y, self.nominal_height]
        return htm

    def _sync(self, reset=False):
        htm = self.htm
        poses = [htm]
        for origin, axis, angle in zip(self.wheel_origins, self.wheel_axes, self._wheel_q):
            spin = np.eye(4)
            spin[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            poses.append(htm @ origin @ spin)
        objects = [self.chassis, *self.wheels, self.base_frame, *self.collision_objects]
        collision_world = [body @ local for body, local in zip(poses, self.collision_poses)]
        poses.extend([htm, *collision_world])
        for obj, pose in zip(objects, poses):
            if reset:
                obj.set_ani_frame(pose)
            else:
                obj.add_ani_frame(self.time, pose)

    def reset(self, base_q=None, wheel_q=None):
        """Reset pose, animation, clock and velocity command."""
        base = _vector(np.zeros(3) if base_q is None else base_q, 3, "base_q")
        wheels = _vector(np.zeros(4) if wheel_q is None else wheel_q, 4, "wheel_q")
        self._base_q, self._wheel_q = base, wheels
        self._command = np.zeros(2)
        self.time = 0.0
        self._sync(reset=True)

    def set_velocity(self, v, w):
        """Set forward speed v and counterclockwise yaw rate w until changed."""
        self._command = _vector([v, w], 2, "[v, w]")

    def step(self, dt):
        """Advance exactly along a constant-command SE(2) arc for dt seconds."""
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        v, w = self._command
        turn = w * dt
        mid_heading = self._base_q[2] + turn / 2
        # np.sinc handles both straight motion and arbitrarily small yaw rates.
        distance = v * dt * np.sinc(turn / (2 * np.pi))
        self._base_q[:2] += distance * np.array([np.cos(mid_heading), np.sin(mid_heading)])
        self._base_q[2] += turn
        speeds = v + np.array([-1, 1, -1, 1]) * w * self.track_width / 2
        self._wheel_q += dt * speeds / self.wheel_radius
        self.time += dt
        self._sync()

    def add_to_simulation(self, sim, *, show_collision=False, show_base_frame=True):
        sim.add([self.chassis, *self.wheels])
        if show_base_frame:
            sim.add(self.base_frame)
        if show_collision:
            sim.add(self.collision_objects)


def create_scout_mini(*, name="scout", base_q=None,
                      asset_url="http://127.0.0.1:8002/models/scout_piper/assets"):
    """Build Scout visuals and Box/Cylinder collision primitives from scene.xml.

    The tire-center track, rather than the offset URDF wheel-joint spacing,
    determines the ideal wheel speeds. All mesh transforms are already baked
    into the project's OBJ files, so their scale remains one.
    """
    scene = ET.parse(MODEL_DIR / "scene.xml")
    body = scene.find(".//body[@name='base_link']")
    wheel_bodies = [body.find(f"body[@name='{wheel}']") for wheel in WHEEL_NAMES]
    origins = tuple(_pose(wheel) for wheel in wheel_bodies)
    chassis_geom = body.find("geom[@name='chassis_contact']")
    wheel_geoms = [wheel.find("geom[@class='collision']") for wheel in wheel_bodies]
    collision_poses = tuple(_pose(geom) for geom in [chassis_geom, *wheel_geoms])
    half_size = np.fromstring(chassis_geom.get("size"), sep=" ")
    collisions = [ub.Box(name=f"{name}_col_chassis", width=2 * half_size[0],
                          depth=2 * half_size[1], height=2 * half_size[2],
                          color="#26a69a", opacity=0.25)]
    radii = []
    for i, geom in enumerate(wheel_geoms):
        radius, half_height = np.fromstring(geom.get("size"), sep=" ")
        radii.append(radius)
        collisions.append(ub.Cylinder(name=f"{name}_col_wheel_{i}", radius=radius,
                                       height=2 * half_height, color="#42a5f5", opacity=0.25))
    centers = [origin @ local for origin, local in zip(origins, collision_poses[1:])]
    track = float(centers[0][1, 3] - centers[1][1, 3])
    model = ScoutMiniModel(
        chassis=ub.RigidObject(_visuals(body, asset_url), name=f"{name}_chassis"),
        wheels=[ub.RigidObject(_visuals(wheel, asset_url), name=f"{name}_{i}")
                for i, wheel in enumerate(wheel_bodies)],
        wheel_origins=origins,
        wheel_axes=np.array([np.fromstring(wheel.find("joint").get("axis"), sep=" ")
                             for wheel in wheel_bodies]),
        collision_objects=collisions, collision_poses=collision_poses,
        base_frame=ub.Frame(name=f"{name}_base_frame", size=0.15),
        nominal_height=float(_pose(body)[2, 3]), wheel_radius=float(radii[0]), track_width=track,
    )
    model.reset(base_q=base_q)
    return model


@dataclass
class ScoutPiperModel:
    scout: ScoutMiniModel
    piper: PiperModel
    mount_htm: np.ndarray

    @property
    def arm_q(self):
        """Six URDF arm angles, independent of uaibot's DH offsets."""
        return np.asarray(self.piper.robot.q).reshape(6).copy() - self.piper.kinematics.dh[:, 0]

    @property
    def time(self):
        return self.scout.time

    def reset(self, base_q=None, arm_q=None, wheel_q=None):
        """Reset the assembly, preserving arm posture if arm_q is omitted."""
        arm = self.arm_q if arm_q is None else _vector(arm_q, 6, "arm_q")
        self.scout.reset(base_q=base_q, wheel_q=wheel_q)
        self.piper.set_configuration(arm, htm=self.scout.htm @ self.mount_htm)

    def set_velocity(self, v, w):
        self.scout.set_velocity(v, w)

    def step(self, dt, *, arm_q=None):
        """Advance the chassis, then rigidly carry the arm; no control is run.

        Optional arm_q prescribes the next URDF joint pose directly. Omitted
        arm_q holds the current joints fixed relative to the moving chassis.
        """
        arm = self.arm_q if arm_q is None else _vector(arm_q, 6, "arm_q")
        self.scout.step(dt)
        self.piper.add_ani_frame(self.time, arm, htm=self.scout.htm @ self.mount_htm)

    def add_to_simulation(self, sim, *, show_collision=False, show_base_frame=True):
        self.scout.add_to_simulation(sim, show_collision=show_collision,
                                     show_base_frame=show_base_frame)
        self.piper.add_to_simulation(sim, show_collision=show_collision)


def create_scout_piper(*, name="scout_piper", base_q=None, arm_q=None,
                       gripper_opening=0.06, show_dh=False,
                       asset_url="http://127.0.0.1:8002/models/scout_piper/assets"):
    """Build Scout Mini with the existing Piper factory and pinned rigid mount.

    Scout base_q is world [x,y,yaw]. The base height and mount transform come
    from the existing model. The six Piper joint coordinates remain unchanged.
    """
    scout = create_scout_mini(name=f"{name}_scout", base_q=base_q, asset_url=asset_url)
    scene = ET.parse(MODEL_DIR / "scene.xml")
    mount = _pose(scene.find(".//body[@name='piper_base_link']"))
    piper = create_piper(name=f"{name}_piper", q0=arm_q, htm=scout.htm @ mount,
                         gripper_opening=gripper_opening, show_dh=show_dh, asset_url=asset_url)
    return ScoutPiperModel(scout, piper, mount)
