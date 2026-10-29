"""Project-local Piper factory; no modifications to uaibot are required.

Distances and angles use meters and radians. DH is the standard convention:
Rz(theta) Tz(d) Tx(a) Rx(alpha). uaibot's revolute q is the *full* DH angle,
so use PiperModel.to_uaibot_q() when passing URDF/encoder angles to its API.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import trimesh
import uaibot as ub
from scipy.spatial.transform import Rotation

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/scout_piper"
HOME_Q = np.array([0.0, 1.0, -1.2, 0.0, 0.2, 0.0])
# Mount + finger-frame origin - half of the finger length (all from URDF/STL).
NOMINAL_TCP_OFFSET = 0.0045 + 0.138 - 0.0765 / 2
COLORS = ["#ff7043", "#26a69a", "#42a5f5", "#ab47bc", "#ffca28", "#ec407a"]


def _origin(joint):
    origin = joint.find("origin")
    h = np.eye(4)
    h[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    ).as_matrix()
    h[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    return h


def _rz(angle):
    h = np.eye(4)
    h[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    return h


@dataclass
class PiperKinematics:
    # Rows: theta_offset, d, alpha, a. Frames refer to q_URDF == 0.
    dh: np.ndarray
    base_0: np.ndarray
    dh_to_link: tuple
    joint_limits: np.ndarray
    joint_origins: tuple


@lru_cache(maxsize=1)
def piper_kinematics():
    """Derive standard DH from the pinned URDF, including its small offsets.

    z_(i-1) is joint i's axis. Common normals define x_i; intersecting
    axes use their cross product. Parallel axes use their separation.
    Keeping the original rounded RPY values avoids silently changing the CAD.
    """
    tree = ET.parse(MODEL_DIR / "reference/normalized_kinematics.urdf")
    joints = [tree.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
    origins = tuple(_origin(joint) for joint in joints)
    urdf_frames = []
    h = np.eye(4)
    for origin in origins:
        h = h @ origin
        urdf_frames.append(h.copy())

    frames = [urdf_frames[0].copy()]
    for i in range(5):
        previous = frames[-1]
        p, z = previous[:3, 3], previous[:3, 2]
        target = urdf_frames[i + 1]
        point, next_z = target[:3, 3], target[:3, 2]
        cross = np.cross(z, next_z)
        if np.linalg.norm(cross) < 1e-10:
            next_p = point
            separation = point - p - z * np.dot(z, point - p)
            x = separation / np.linalg.norm(separation)
        else:
            s, t = np.linalg.lstsq(
                np.column_stack((z, -next_z)), point - p, rcond=None
            )[0]
            next_p = point + t * next_z
            separation = next_p - (p + s * z)
            if np.linalg.norm(separation) > 1e-10:
                x = separation / np.linalg.norm(separation)
            else:
                x = cross / np.linalg.norm(cross)
        frame = np.eye(4)
        frame[:3, :3] = np.column_stack((x, np.cross(next_z, x), next_z))
        frame[:3, 3] = next_p
        frames.append(frame)
    frames.append(urdf_frames[-1].copy())

    dh = []
    for before, after in pairwise(frames):
        relative = np.linalg.inv(before) @ after
        theta = np.arctan2(relative[1, 0], relative[0, 0])
        alpha = np.arctan2(relative[2, 1], relative[2, 2])
        a = np.dot(relative[:2, 3], [np.cos(theta), np.sin(theta)])
        dh.append([theta, relative[2, 3], alpha, a])
    return PiperKinematics(
        dh=np.array(dh), base_0=frames[0],
        dh_to_link=tuple(np.linalg.inv(frames[i + 1]) @ urdf_frames[i] for i in range(6)),
        joint_limits=np.array([
            [float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))]
            for j in joints
        ]), joint_origins=origins,
    )


def _primitive_collision(link_name, name, color, opacity):
    """Return a basic primitive and its pose in the source URDF link frame.

    Cylinders cover motor housings, base and flanges. Boxes cover elongated
    housings and gripper parts. Each encloses the STL hull with 0.5 mm margin.
    """
    vertices = trimesh.load_mesh(
        MODEL_DIR / f"assets/collision/{link_name}_hull.stl"
    ).vertices
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    size = upper - lower + 0.001
    pose = np.eye(4)
    pose[:3, 3] = center
    if link_name in {"piper_base_link", "link1", "link4", "link6", "flange_link"}:
        radius = float(np.linalg.norm(vertices[:, :2] - center[:2], axis=1).max()) + 0.0005
        obj = ub.Cylinder(name=name, radius=radius, height=float(size[2]),
                          color=color, opacity=opacity)
    else:
        obj = ub.Box(name=name, width=float(size[0]), depth=float(size[1]),
                     height=float(size[2]), color=color, opacity=opacity)
    return obj, pose


@dataclass
class PiperModel:
    """Robot plus its fixed-base collision object and DH coordinate conversion."""

    robot: ub.Robot
    base_collision: ub.Cylinder
    base_collision_pose: np.ndarray
    kinematics: PiperKinematics
    dh_frames: list
    eef_frame: ub.Frame

    def to_uaibot_q(self, q):
        """Convert six URDF joint angles to uaibot's full DH angles."""
        angles = np.asarray(q, dtype=float).reshape(-1)
        if angles.shape != (6,) or not np.all(np.isfinite(angles)):
            raise ValueError("q must contain six finite joint angles in radians")
        return angles + self.kinematics.dh[:, 0]

    def set_configuration(self, q, htm=None):
        """Set a static pose with URDF angles; also update the fixed base."""
        self.robot.set_ani_frame(q=self.to_uaibot_q(q), htm=htm)
        self.base_collision.set_ani_frame(self.robot.htm @ self.base_collision_pose)
        self.eef_frame.set_ani_frame(self.robot.fkm())
        poses = [self.robot.htm @ self.kinematics.base_0, *self.robot.fkm(axis="dh")]
        for frame, pose in zip(self.dh_frames, poses):
            frame.set_ani_frame(pose)

    def add_ani_frame(self, time, q, htm=None):
        """Animate using URDF angles and synchronize TCP, DH and collision poses."""
        self.robot.add_ani_frame(time, self.to_uaibot_q(q), htm=htm)
        self.base_collision.add_ani_frame(time, self.robot.htm @ self.base_collision_pose)
        self.eef_frame.add_ani_frame(time, self.robot.fkm())
        poses = [self.robot.htm @ self.kinematics.base_0, *self.robot.fkm(axis="dh")]
        for frame, pose in zip(self.dh_frames, poses):
            frame.add_ani_frame(time, pose)

    def add_to_simulation(self, sim, show_collision=True):
        sim.add(self.robot)
        sim.add(self.dh_frames)
        sim.add(self.eef_frame)
        if show_collision:
            self.robot.update_col_object(0)
            self.robot.add_col_object(sim)
            sim.add(self.base_collision)


def create_piper(*, name="piper", htm=None, q0=None, gripper_opening=0.06,
                 show_dh=True, opacity=1.0, collision_opacity=0.25, tcp_offset=NOMINAL_TCP_OFFSET,
                 asset_url="http://127.0.0.1:8000/models/scout_piper/assets"):
    """Create the standard six-axis Piper with a fixed-opening gripper.

    Returns PiperModel, whose .robot is a native uaibot.Robot supporting fkm,
    jac_geo, ikm and compute_dist. q0 is in URDF coordinates. The nominal TCP
    is centered in the gripping region between the fingers, 0.10425 m along
    flange +z (4.5 mm mount + 138 mm finger origin - half the 76.5 mm finger
    length from the STL). Its +z is the approach direction;
    +y is the jaw closing axis and +x completes the right-handed frame.
    tcp_offset can override the nominal flange-to-TCP distance for calibration.
    gripper_opening is total gap [0, .1] m.

    Collision objects are basic uaibot cylinders and boxes enclosing the
    project's STL hulls with a 0.5 mm margin, attached to each DH
    link (including flange and gripper on link 6). The fixed base is separate:
    robot.compute_dist covers moving links; use base_collision.compute_dist
    for the base. asset_url must serve the existing visual/ OBJ files over HTTP
    when exporting/viewing the simulation. Construction itself needs no server.
    """
    if not np.isfinite(gripper_opening) or not 0 <= gripper_opening <= 0.1:
        raise ValueError("gripper_opening must be in [0, 0.1] meters")
    if not np.isfinite(tcp_offset) or tcp_offset < 0:
        raise ValueError("tcp_offset must be a finite nonnegative distance")
    kin = piper_kinematics()
    scene = ET.parse(MODEL_DIR / "scene.xml")
    joints = ET.parse(MODEL_DIR / "reference/normalized_kinematics.urdf")

    def visuals(link_name, transform):
        body = scene.find(f".//body[@name='{link_name}']")
        models = []
        for geom in body.findall("geom[@class='visual']"):
            rgba = np.fromstring(geom.get("rgba"), sep=" ")
            color = "#" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])
            mesh_name = geom.get("mesh")
            models.append(ub.Model3D(
                f"{asset_url.rstrip('/')}/visual/{mesh_name}.obj", htm=transform,
                mesh_material=ub.MeshMaterial(color=color, opacity=opacity,
                                               roughness=0.6, metalness=0.15),
            ))
        return models

    links = []
    for i, (theta, d, alpha, a) in enumerate(kin.dh):
        transform = kin.dh_to_link[i]
        # Revolute offsets live entirely in q: Python ignores Link.theta, but
        # the JS renderer applies it to the mesh hierarchy. Keep it zero so
        # both backends represent the same geometry. Explicit Frame objects
        # also avoid missing DH axes in this version's JS show_frame path.
        link = ub.Link(i, 0.0, d, alpha, a, 0,
                       visuals(f"link{i + 1}", transform), show_frame=False)
        collider, primitive_pose = _primitive_collision(
            f"link{i + 1}", f"{name}_col_link{i + 1}", COLORS[i], collision_opacity)
        # Uaibot 1.2.7's C++ conversion requires column-preserving matrix slices.
        link.attach_col_object(collider, np.asmatrix(transform @ primitive_pose))
        links.append(link)

    # Branching gripper is rigid at the chosen opening, attached to the wrist.
    fixed_frames = {"link6": np.eye(4)}
    for joint_name, displacement in [
        ("flange_joint", 0), ("gripper_base_joint", 0),
        ("gripper_joint1", gripper_opening / 2),
        ("gripper_joint2", -gripper_opening / 2),
    ]:
        joint = joints.find(f"joint[@name='{joint_name}']")
        local = _origin(joint)
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ") if joint.find("axis") is not None else np.zeros(3)
        translation = np.eye(4)
        translation[:3, 3] = displacement * axis
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        fixed_frames[child] = fixed_frames[parent] @ local @ translation
        transform = kin.dh_to_link[-1] @ fixed_frames[child]
        links[-1].list_model_3d.extend(visuals(child, transform))
        collider, primitive_pose = _primitive_collision(
            child, f"{name}_col_{child}", COLORS[-1], collision_opacity)
        links[-1].attach_col_object(collider, np.asmatrix(transform @ primitive_pose))

    q0 = HOME_Q if q0 is None else np.asarray(q0, dtype=float).reshape(-1)
    if q0.shape != (6,) or not np.all(np.isfinite(q0)):
        raise ValueError("q0 must contain six finite joint angles in radians")
    flange_tcp = np.eye(4)
    flange_tcp[2, 3] = tcp_offset
    robot = ub.Robot(
        name=name, links=links, list_base_3d_obj=visuals("piper_base_link", np.eye(4)),
        htm=np.eye(4) if htm is None else htm, htm_base_0=np.asmatrix(kin.base_0),
        htm_n_eef=np.asmatrix(kin.dh_to_link[-1] @ flange_tcp), q0=q0 + kin.dh[:, 0],
        joint_limits=kin.joint_limits + kin.dh[:, 0, None], eef_frame_visible=False,
    )
    base, base_pose = _primitive_collision("piper_base_link", f"{name}_col_base", "#78909c", collision_opacity)
    base.set_ani_frame(robot.htm @ base_pose)
    poses = [robot.htm @ kin.base_0, *robot.fkm(axis="dh")]
    frames = [ub.Frame(name=f"{name}_dh_{i}", htm=pose, size=0.075)
              for i, pose in enumerate(poses)] if show_dh else []
    eef_frame = ub.Frame(name=f"{name}_tcp", htm=robot.fkm(), size=0.075,
                         axis_color=["#e53935", "#43a047", "#1565ff"])
    return PiperModel(robot, base, base_pose, kin, frames, eef_frame)
