#!/usr/bin/env python3
"""Rebuild the standalone robot from checksum-pinned official AgileX inputs."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import collada
import numpy as np
import trimesh
from arm_dynamics import configure_arm_dynamics
from scipy.spatial.transform import Rotation
from upstream import CACHE, ensure_sources

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models/scout_piper"
SCOUT = CACHE / "scout_ros/scout_description"
PIPER = CACHE / "agx_arm_urdf"
HOME_ARM = [0.0, 0.7, -1.4, 0.0, 0.7, 0.0]


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in np.asarray(values).ravel())


def vector(element, attribute, default="0 0 0"):
    return np.fromstring(
        element.get(attribute, default) if element is not None else default, sep=" "
    )


def pose(origin):
    result = {"pos": numbers(vector(origin, "xyz"))}
    rpy = vector(origin, "rpy")
    xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    result["quat"] = numbers(xyzw[[3, 0, 1, 2]])
    return result


def tensor(element):
    a = element.attrib
    return np.array(
        [
            [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
            [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
            [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
        ]
    )


def tensor6(matrix):
    return matrix[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]


def audit(mass, center, inertia, bounds):
    """Necessary rigid-body checks, including bounded variance around the stated COM."""
    principal = np.linalg.eigvalsh(inertia)
    reasons = []
    if mass <= 0 or principal[0] <= 0:
        reasons.append("nonpositive mass or principal inertia")
    if principal[-1] > principal[:2].sum() + 1e-12:
        reasons.append("principal moments violate triangle inequality")
    lo, hi = bounds
    if np.any(center < lo - 1e-7) or np.any(center > hi + 1e-7):
        reasons.append("COM outside mesh bounding box")
    variance = (np.trace(inertia) / 2 - np.diag(inertia)) / mass
    limit = (center - lo) * (hi - center)
    if np.any(variance > limit + 1e-9):
        reasons.append("COM variance exceeds mesh support bound")
    return reasons, variance, limit


def write_obj(path, primitives):
    """Bake Collada scene transforms; preserve triangle winding and split normals."""
    with path.open("w") as out:
        out.write(
            "# Converted from pinned official AgileX Collada; meters. See source_manifest.json.\n"
        )
        voffset = noffset = 1
        for vertices, faces, normals, normal_faces in primitives:
            np.savetxt(out, vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(out, normals, fmt="vn %.9g %.9g %.9g")
            for face, normal_face in zip(faces + voffset, normal_faces + noffset):
                out.write(
                    "f "
                    + " ".join(f"{v}//{n}" for v, n in zip(face, normal_face))
                    + "\n"
                )
            voffset += len(vertices)
            noffset += len(normals)


def visual_assets(name, source_path, rotation, asset, manifest):
    document = collada.Collada(str(source_path))
    if document.assetInfo.unitmeter != 1.0 or document.assetInfo.upaxis != "Z_UP":
        raise ValueError(f"Unexpected source units/axes: {source_path}")
    groups = {}
    vertices_all = []
    tire = []
    for geometry in document.scene.objects("geometry"):
        for primitive in geometry.primitives():
            if hasattr(primitive, "triangleset"):
                primitive = primitive.triangleset()
            vertices = np.asarray(primitive.vertex, dtype=float) @ rotation.T
            normals = np.asarray(primitive.normal, dtype=float) @ rotation.T
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            color = tuple(float(x) for x in primitive.material.effect.diffuse)
            groups.setdefault(color, []).append(
                (vertices, primitive.vertex_index, normals, primitive.normal_index)
            )
            vertices_all.append(vertices)
            if name == "wheel" and np.allclose(color[:3], [0.2941176593] * 3):
                tire.append(vertices)
    visual_geoms = []
    if name == "wheel":
        measured = np.concatenate(tire)
        radial_center = (measured.min(0) + measured.max(0)) / 2
        shift = np.array([-radial_center[0], -radial_center[1], 0])
        # The CAD mesh's tire is about 1 mm off the named axle. Center all its
        # parts together; do not turn an export offset into rotating eccentricity.
        for vertices in vertices_all:
            vertices += shift
        manifest["wheel_visual_correction"] = {
            "translation_m": shift.tolist(),
            "class": 3,
            "method": "center the tire radial bounding box on the wheel rotation axis",
        }
    for index, (color, primitives) in enumerate(groups.items()):
        mesh_name = f"{name}_visual_{index}"
        relative = f"assets/visual/{mesh_name}.obj"
        write_obj(OUTPUT / relative, primitives)
        ET.SubElement(asset, "mesh", name=mesh_name, file=relative, inertia="shell")
        visual_geoms.append(
            {
                "name": mesh_name,
                "type": "mesh",
                "mesh": mesh_name,
                "rgba": numbers(color),
                "class": "visual",
            }
        )
        manifest["assets"][relative] = {
            "source": str(source_path.relative_to(ROOT)),
            "diffuse_rgba": color,
            "triangles": sum(len(p[1]) for p in primitives),
        }
    return (
        visual_geoms,
        np.concatenate(vertices_all),
        np.concatenate(tire) if tire else None,
    )


def roof_heights(vertices, faces, xy):
    triangles = vertices[faces]
    a = triangles[:, 0, :2]
    u = triangles[:, 1, :2] - a
    v = triangles[:, 2, :2] - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    good = abs(det) > 1e-12
    r = xy - a
    b = np.divide(
        r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0], det, out=np.zeros_like(det), where=good
    )
    c = np.divide(
        u[:, 0] * r[:, 1] - u[:, 1] * r[:, 0], det, out=np.zeros_like(det), where=good
    )
    mask = good & (b >= -1e-8) & (c >= -1e-8) & (b + c <= 1 + 1e-8)
    z = (
        triangles[:, 0, 2]
        + b * (triangles[:, 1, 2] - triangles[:, 0, 2])
        + c * (triangles[:, 2, 2] - triangles[:, 0, 2])
    )
    return float(z[mask].max())


def source_models():
    """Expand only the four-wheel Mini macro, without requiring ROS or Xacro."""
    scout_path = SCOUT / "urdf/scout_mini.urdf.xacro"
    wheel_path = SCOUT / "urdf/scout_mini_wheel.xacro"
    source = ET.parse(scout_path).getroot()
    wheel_macro = (
        ET.parse(wheel_path).getroot().find("{http://ros.org/wiki/xacro}macro")
    )
    props = {
        e.get("name"): float(e.get("value"))
        for e in source.findall("{http://ros.org/wiki/xacro}property")
    }
    scout = ET.Element("robot", name="scout_mini")
    base = copy.deepcopy(source.find("link[@name='base_link']"))
    base.append(copy.deepcopy(source.find("link[@name='inertial_link']/inertial")))
    scout.append(base)
    # Normalize rounded +/-1.57 rotations to exact right angles. Positive rotation
    # is forward rolling on every wheel; the left source joint coordinates flip sign.
    for side in ["front_left", "front_right", "rear_left", "rear_right"]:
        left = side.endswith("left")
        front = side.startswith("front")
        link = copy.deepcopy(wheel_macro.find("link"))
        link.set("name", f"{side}_wheel_link")
        scout.append(link)
        j = ET.SubElement(scout, "joint", name=f"{side}_wheel", type="continuous")
        ET.SubElement(j, "parent", link="base_link")
        ET.SubElement(j, "child", link=link.get("name"))
        ET.SubElement(
            j,
            "origin",
            xyz=numbers(
                [
                    (1 if front else -1) * props["wheelbase"] / 2,
                    (1 if left else -1) * props["track"] / 2,
                    props["wheel_vertical_offset"],
                ]
            ),
            rpy=numbers([-np.pi / 2 if left else np.pi / 2, 0, 0]),
        )
        ET.SubElement(j, "axis", xyz="0 0 1" if left else "0 0 -1")
    arm_path = PIPER / "piper/urdf/piper_description.urdf"
    gripper_path = PIPER / "piper/urdf/piper_with_gripper_description.xacro"
    arm = ET.parse(arm_path).getroot()
    arm.remove(arm.find("link[@name='world']"))
    arm.remove(arm.find("joint[@name='world_to_base_link']"))
    arm.find("link[@name='base_link']").set("name", "piper_base_link")
    arm.find("joint[@name='joint1']/parent").set("link", "piper_base_link")
    gripper = ET.parse(gripper_path).getroot()
    return (
        scout,
        arm,
        gripper,
        [
            (scout_path, "scout_mini.urdf.xacro"),
            (wheel_path, "scout_mini_wheel.xacro"),
            (arm_path, "piper_description.urdf"),
            (gripper_path, "piper_with_gripper_description.xacro"),
        ],
    )


def scene_triangles(path, rotation):
    vertices, faces, offset = [], [], 0
    for g in collada.Collada(str(path)).scene.objects("geometry"):
        for p in g.primitives():
            if hasattr(p, "triangleset"):
                p = p.triangleset()
            vertices.append(np.asarray(p.vertex) @ rotation.T)
            faces.append(p.vertex_index + offset)
            offset += len(p.vertex)
    return np.concatenate(vertices), np.concatenate(faces)


def build(fetch=False):
    lock = ensure_sources(fetch)
    config = json.loads((OUTPUT / "build_config.json").read_text())
    # Read literal manufacturer SDK coefficients without executing external Python.
    constants = ast.parse((CACHE / "pyAgxArm/pyAgxArm/api/constants.py").read_text())
    coefficients = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in constants.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("ROBOT_JOINT_TORQUE_B", "ROBOT_JOINT_TORQUE_C")
    }
    sdk_limits = [
        8 * b * c
        for b, c in zip(
            coefficients["ROBOT_JOINT_TORQUE_B"]["piper"],
            coefficients["ROBOT_JOINT_TORQUE_C"]["piper"],
        )
    ]
    if not np.allclose(
        config["arm"]["torque_limits_Nm"], sdk_limits, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Torque config differs from the selected standard-Piper DEFAULT SDK profile"
        )
    for subdir in ["assets/visual", "assets/collision", "licenses", "reference"]:
        (OUTPUT / subdir).mkdir(parents=True, exist_ok=True)
    scout, arm, gripper, references = source_models()
    links = {
        e.get("name"): e for tree in [scout, arm, gripper] for e in tree.findall("link")
    }
    paths = {
        name: references[0][0]
        if name == "base_link"
        else references[1][0]
        if name.endswith("_wheel_link")
        else references[2][0]
        if name in {e.get("name") for e in arm.findall("link")}
        else references[3][0]
        for name in links
    }
    manifest = {
        "description": "Rebuilt from pinned official Scout Mini and standard Piper sources.",
        "build_toolchain": {
            name: version(name) for name in ["numpy", "scipy", "trimesh", "pycollada"]
        },
        "classifications": config["classification"],
        "repositories": lock["repositories"],
        "manuals": lock["manuals"],
        "source_files": {},
        "assets": {},
        "inertials": {},
        "kinematics": {},
        "reference_files": {},
        "assumptions": config,
        "legacy_workspace": lock["legacy_workspace"],
        "investigated_repositories": lock["investigated_repositories"],
        "mass_summary": {"scout_kg": config["scout"]["kerb_mass_kg"]},
    }
    for key, repo in lock["repositories"].items():
        for relative, digest in repo["files"].items():
            manifest["source_files"][
                str((CACHE / key / relative).relative_to(ROOT))
            ] = digest
    root = ET.Element("mujoco", model="Scout Mini with standard Piper")
    root.append(
        ET.Comment(
            " Generated from pinned official sources. See docs/physical_model_report.md. "
        )
    )
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        autolimits="true",
        inertiafromgeom="false",
        fusestatic="false",
    )
    ET.SubElement(
        root,
        "option",
        timestep=numbers([config["simulation"]["timestep_s"]]),
        gravity=numbers(config["simulation"]["gravity"]),
        integrator="implicitfast",
        cone="elliptic",
    )
    ET.SubElement(root, "statistic", center="0 0 0.45", extent="1.3")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        azimuth="135",
        elevation="-20",
        offwidth="1440",
        offheight="1080",
    )
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3")
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default, "joint", damping="0", frictionloss="0", stiffness="0", armature="0"
    )
    ET.SubElement(
        default, "geom", friction=numbers(config["simulation"]["contact_friction"])
    )
    for class_name, attrs in [
        ("visual", {"contype": "0", "conaffinity": "0", "group": "2", "density": "0"}),
        ("collision", {"group": "3", "rgba": "0.2 0.7 0.3 0.3", "density": "0"}),
    ]:
        ET.SubElement(
            ET.SubElement(default, "default", {"class": class_name}), "geom", attrs
        )
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="key_light",
        pos="0 -2 3",
        dir="0 0.5 -1",
        directional="true",
    )
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="5 5 0.1",
        rgba="0.82 0.85 0.88 1",
    )
    rotation = Rotation.from_euler(
        "xyz", vector(links["base_link"].find("visual/origin"), "rpy")
    ).as_matrix()
    converted = {}
    mesh_files = {}
    for name in [
        "base_link",
        "wheel",
        *[e.get("name") for e in arm.findall("link")],
        "flange_link",
        "gripper_base",
        "gripper_link1",
        "gripper_link2",
    ]:
        link = links["front_left_wheel_link"] if name == "wheel" else links[name]
        mesh_uri = link.find("visual/geometry/mesh").get("filename")
        if "agx_arm_urdf/" in mesh_uri:
            source_path = PIPER / mesh_uri.split("agx_arm_urdf/")[1]
            physical_path = (
                PIPER
                / link.find("collision/geometry/mesh")
                .get("filename")
                .split("agx_arm_urdf/")[1]
            )
        else:
            source_path = SCOUT / mesh_uri.split("scout_description/")[1]
            physical_path = source_path
        mesh_files[name] = source_path
        vg, vertices, tire = visual_assets(
            name,
            source_path,
            rotation if name == "base_link" else np.eye(3),
            asset,
            manifest,
        )
        physical = (
            vertices
            if source_path == physical_path
            else trimesh.load_mesh(physical_path).vertices
        )
        hull = trimesh.convex.convex_hull(physical)
        if name not in ("base_link", "wheel"):
            relative = f"assets/collision/{name}_hull.stl"
            hull.export(OUTPUT / relative)
            ET.SubElement(asset, "mesh", name=f"{name}_hull", file=relative)
            manifest["assets"][relative] = {
                "source": str(physical_path.relative_to(ROOT)),
                "operation": "convex hull",
                "triangles": len(hull.faces),
            }
        converted[name] = (vg, vertices, tire, hull)
    # Re-evaluate flush mounting against the new official chassis, including its source rotation.
    chassis_vertices, chassis_faces = scene_triangles(mesh_files["base_link"], rotation)
    base_bounds = converted["piper_base_link"][3].bounds
    lo, hi = base_bounds
    heights = [
        roof_heights(chassis_vertices, chassis_faces, [x, y])
        for x in np.linspace(lo[0], hi[0], 11)
        for y in np.linspace(lo[1], hi[1], 11)
    ]
    triangles = chassis_vertices[chassis_faces]
    overlaps = np.all(triangles[:, :, :2].max(1) >= lo[:2], axis=1) & np.all(
        triangles[:, :, :2].min(1) <= hi[:2], axis=1
    )
    conservative_top = float(triangles[overlaps, :, 2].max())
    mount = np.array([0, 0, np.ceil((conservative_top - lo[2]) * 1e5) / 1e5])
    manifest["mount"] = {
        "xyz": mount.tolist(),
        "rpy": [0, 0, 0],
        "class": 3,
        "piper_base_bounds_m": base_bounds.tolist(),
        "deck_grid": [11, 11],
        "deck_z_range_m": [min(heights), max(heights)],
        "conservative_deck_z_max_m": conservative_top,
        "min_clearance_m": float(mount[2] + lo[2] - max(heights)),
    }
    tire = converted["wheel"][2]
    tire_center = (tire.min(0) + tire.max(0)) / 2
    radius, halfwidth = (
        config["scout"]["wheel_radius_m"],
        config["scout"]["wheel_width_m"] / 2,
    )
    manifest["wheel_collision"] = {
        "pos": tire_center.tolist(),
        "radius": radius,
        "halfwidth": halfwidth,
        "class": 3,
        "source": "official simulator cylinder dimensions; measured tire-envelope center",
    }
    wheel_joints = scout.findall("joint")
    source_wheel_origins = {
        j.get("name"): vector(j.find("origin"), "xyz").tolist() for j in wheel_joints
    }
    # Manufacturer drawing outranks the older axle layout. Derive axle-joint
    # track from tire-center track and the measured axial mesh offset.
    manual = config["manufacturer_cross_checks"]["scout_mini"]
    for j in wheel_joints:
        name = j.get("name")
        original = vector(j.find("origin"), "xyz")
        xyz = [
            manual["wheelbase_m"] / 2 * (1 if name.startswith("front") else -1),
            (manual["tire_track_m"] / 2 - abs(tire_center[2]))
            * (1 if name.endswith("left_wheel") else -1),
            original[2],
        ]
        j.find("origin").set("xyz", numbers(xyz))
    assembled_vertices = [converted["base_link"][1]]
    for j in wheel_joints:
        R = Rotation.from_euler("xyz", vector(j.find("origin"), "rpy")).as_matrix()
        assembled_vertices.append(
            converted["wheel"][1] @ R.T + vector(j.find("origin"), "xyz")
        )
    assembled_vertices = np.concatenate(assembled_vertices)
    manifest["scout_visual_envelope"] = {
        "class": 3,
        "dimensions_m": np.ptp(assembled_vertices, axis=0).tolist(),
        "min_m": assembled_vertices.min(0).tolist(),
        "max_m": assembled_vertices.max(0).tolist(),
        "note": "manual in-plane wheel placement; original URDF vertical stance retained",
    }
    bottoms = []
    for j in wheel_joints:
        R = Rotation.from_euler("xyz", vector(j.find("origin"), "rpy")).as_matrix()
        center = vector(j.find("origin"), "xyz") + R @ tire_center
        axial_z = abs(R[2, 2])
        bottoms.append(
            center[2] - radius * np.sqrt(1 - axial_z**2) - halfwidth * axial_z
        )
    base_height = float(-min(bottoms) + 0.001)
    base = ET.SubElement(
        world, "body", name="base_link", pos=numbers([0, 0, base_height]), gravcomp="0"
    )
    ET.SubElement(base, "freejoint", name="floating_base")
    bodies = {"base_link": base}

    def populate(name, body):
        key = "wheel" if name.endswith("_wheel_link") else name
        vg, vertices, _, hull = converted[key]
        for attrs in vg:
            attrs = dict(attrs)
            attrs["name"] = name + "_" + attrs["name"]
            ET.SubElement(body, "geom", attrs)
        if key == "wheel":
            ET.SubElement(
                body,
                "geom",
                {
                    "name": name + "_contact",
                    "class": "collision",
                    "type": "cylinder",
                    "pos": numbers(tire_center),
                    "size": numbers([radius, halfwidth]),
                },
            )
        elif key == "base_link":
            low, high = vertices.min(0), vertices.max(0)
            high[2] = mount[2]
            ET.SubElement(
                body,
                "geom",
                {
                    "name": "chassis_contact",
                    "class": "collision",
                    "type": "box",
                    "pos": numbers((low + high) / 2),
                    "size": numbers((high - low) / 2),
                },
            )
        else:
            ET.SubElement(
                body,
                "geom",
                {
                    "name": name + "_contact",
                    "class": "collision",
                    "type": "mesh",
                    "mesh": name + "_hull",
                },
            )
        ine = links[name].find("inertial")
        source_mass = float(ine.find("mass").get("value"))
        center = vector(ine.find("origin"), "xyz")
        R = Rotation.from_euler("xyz", vector(ine.find("origin"), "rpy")).as_matrix()
        inertia = R @ tensor(ine.find("inertia")) @ R.T
        reasons, variance, bound = audit(source_mass, center, inertia, hull.bounds)
        record = {
            "source": str(paths[name].relative_to(ROOT)),
            "source_mass_kg": source_mass,
            "source_com_m": center.tolist(),
            "source_tensor_kg_m2": inertia.tolist(),
            "source_mesh_bounds_m": hull.bounds.tolist(),
            "source_variance_m2": variance.tolist(),
            "variance_bound_m2": bound.tolist(),
            "rejection_reasons": reasons,
            "source_mass_class": 1 if name == "link6" else 2,
        }
        mass = source_mass
        if key == "base_link":
            mass = (
                config["scout"]["kerb_mass_kg"] - 4 * config["scout"]["wheel_mass_kg"]
            )
            if mass <= 0:
                raise ValueError("Nonpositive chassis residual mass")
            center = hull.center_mass
            inertia = hull.moment_inertia * (mass / hull.mass)
            record.update(
                mass_class=4,
                com_class=3,
                inertia_class=3,
                method="uniform chassis hull at kerb mass minus four official/unvalidated wheel masses",
            )
        elif key == "wheel":
            mass = config["scout"]["wheel_mass_kg"]
            center = tire_center.copy()
            transverse = mass * (3 * radius**2 + (2 * halfwidth) ** 2) / 12
            inertia = np.diag([transverse, transverse, mass * radius**2 / 2])
            record.update(
                mass_class=2,
                com_class=3,
                inertia_class=3,
                method="uniform solid cylinder; mass not independently validated",
            )
        elif reasons:
            center = hull.center_mass
            inertia = hull.moment_inertia * (mass / hull.mass)
            record.update(
                mass_class=record["source_mass_class"],
                com_class=3,
                inertia_class=3,
                method="uniform-density source collision convex hull at official mass",
            )
        else:
            record.update(
                mass_class=2,
                com_class=2,
                inertia_class=2,
                method="official URDF COM and full tensor retained after necessary consistency checks",
            )
        support = hull.bounds.copy()
        if key == "wheel":
            support = np.array(
                [
                    center - [radius, radius, halfwidth],
                    center + [radius, radius, halfwidth],
                ]
            )
        record.update(
            mass_kg=mass,
            com_m=center.tolist(),
            tensor_kg_m2=inertia.tolist(),
            mesh_bounds_m=support.tolist(),
        )
        if audit(mass, center, inertia, support)[0]:
            raise ValueError(f"Invalid final inertia: {name}")
        manifest["inertials"][name] = record
        ET.SubElement(
            body,
            "inertial",
            mass=numbers([mass]),
            pos=numbers(center),
            fullinertia=numbers(tensor6(inertia)),
        )

    def add_joint(j):
        name = j.get("name")
        child = j.find("child").get("link")
        parent = j.find("parent").get("link")
        body = ET.SubElement(
            bodies[parent], "body", name=child, gravcomp="0", **pose(j.find("origin"))
        )
        bodies[child] = body
        info = {
            "parent": parent,
            "child": child,
            "type": j.get("type"),
            "xyz": vector(j.find("origin"), "xyz").tolist(),
            "rpy": vector(j.find("origin"), "rpy").tolist(),
            "class": 2,
        }
        if name.endswith("_wheel"):
            info.update(
                classification_override={
                    "x": 1,
                    "y": 3,
                    "z": 2,
                    "rpy": 4,
                    "axis_sign": 4,
                },
                source_xyz=source_wheel_origins[name],
            )
        if j.get("type") != "fixed":
            attrs = {
                "name": name,
                "type": "slide" if j.get("type") == "prismatic" else "hinge",
                "axis": j.find("axis").get("xyz"),
            }
            info["axis"] = vector(j.find("axis"), "xyz").tolist()
            limit = j.find("limit")
            if limit is not None:
                attrs["range"] = f"{limit.get('lower')} {limit.get('upper')}"
                info["range"] = [float(limit.get("lower")), float(limit.get("upper"))]
                info["urdf_effort"] = float(limit.get("effort"))
                info["urdf_velocity"] = float(limit.get("velocity"))
            ET.SubElement(body, "joint", attrs)
        manifest["kinematics"][name] = info
        populate(child, body)

    populate("base_link", base)
    for j in wheel_joints:
        add_joint(j)
    mount_joint = ET.fromstring(
        '<joint name="scout_to_piper_joint" type="fixed"><parent link="base_link"/><child link="piper_base_link"/><origin rpy="0 0 0"/></joint>'
    )
    mount_joint.find("origin").set("xyz", numbers(mount))
    add_joint(mount_joint)
    manifest["kinematics"]["scout_to_piper_joint"]["class"] = 3
    for j in arm.findall("joint"):
        add_joint(j)
    for j in gripper.findall("joint"):
        if j.get("name") != "gripper":
            add_joint(j)
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "joint",
        name="gripper_symmetry",
        joint1="gripper_joint2",
        joint2="gripper_joint1",
        polycoef="0 -1 0 0 0",
        solref=numbers(config["simulation"]["gripper_equality_solref"]),
        solimp=numbers(config["simulation"]["gripper_equality_solimp"]),
    )
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="home",
        qpos=numbers(
            [0, 0, base_height, 1, 0, 0, 0, *([0] * 4), *HOME_ARM, 0.02, -0.02]
        ),
    )
    params = {}
    for i, j in enumerate(arm.findall("joint")):
        params[j.get("name")] = {
            "effort_Nm": float(j.find("limit").get("effort")),
            "command_limit_Nm": config["arm"]["torque_limits_Nm"][i],
            "damping_Nm_s_per_rad": 0.0,
            "frictionloss_Nm": 0.0,
            "profile": config["arm"]["torque_profile"],
        }
    manifest["arm_dynamics"] = configure_arm_dynamics(root, params)
    manifest["initial_base_height_m"] = base_height
    manifest["home_arm_rad"] = HOME_ARM
    manifest["wheel_frames"] = {
        "position_classes": {"x": 1, "y": 3, "z": 2},
        "normalization_class": 4,
        "source_origins": source_wheel_origins,
        "description": "manual wheelbase and tire-center track; retain URDF vertical stance; exact +/-pi/2 wheel rotations; positive spin forward on all wheels",
    }
    manifest["mass_summary"].update(
        chassis_kg=manifest["inertials"]["base_link"]["mass_kg"],
        wheels_kg=sum(
            v["mass_kg"]
            for n, v in manifest["inertials"].items()
            if n.endswith("_wheel_link")
        ),
        arm_with_flange_kg=sum(
            manifest["inertials"][n]["mass_kg"]
            for n in [
                "piper_base_link",
                *[f"link{i}" for i in range(1, 7)],
                "flange_link",
            ]
        ),
        gripper_kg=sum(
            manifest["inertials"][n]["mass_kg"]
            for n in ["gripper_base", "gripper_link1", "gripper_link2"]
        ),
        robot_kg=sum(v["mass_kg"] for v in manifest["inertials"].values()),
    )
    # Store small source snapshots for independent FK validation without network/cache/ROS.
    for src, dest in references:
        target = OUTPUT / "reference" / dest
        shutil.copyfile(src, target)
        manifest["reference_files"]["reference/" + dest] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
    canonical = ET.Element("robot", name="normalized_scout_piper")
    for tree in [scout, arm, gripper]:
        for j in tree.findall("joint"):
            if j.get("name") != "gripper":
                canonical.append(copy.deepcopy(j))
    canonical.insert(4, mount_joint)
    ET.indent(canonical, space="  ")
    target = OUTPUT / "reference/normalized_kinematics.urdf"
    ET.ElementTree(canonical).write(target, encoding="utf-8", xml_declaration=True)
    manifest["reference_files"]["reference/normalized_kinematics.urdf"] = (
        hashlib.sha256(target.read_bytes()).hexdigest()
    )
    shutil.copyfile(PIPER / "LICENSE", OUTPUT / "licenses/Piper-MIT.txt")
    bsd = (
        (SCOUT / "urdf/scout_wheel_type1.xacro")
        .read_text()
        .split("<!--", 1)[1]
        .split("-->", 1)[0]
    )
    (OUTPUT / "licenses/Scout-BSD.txt").write_text(bsd.strip() + "\n")
    for relative, record in manifest["assets"].items():
        record["sha256"] = hashlib.sha256((OUTPUT / relative).read_bytes()).hexdigest()
        source = ROOT / record["source"]
        key, relative_input = (
            source.relative_to(CACHE).parts[0],
            "/".join(source.relative_to(CACHE).parts[1:]),
        )
        repo = lock["repositories"][key]
        record["source_url"] = (
            f"{repo['repository']}/blob/{repo['commit']}/{relative_input}"
        )
        record["geometry_class"] = 2 if "/visual/" in relative else 3
    # Remove stale generated meshes only; reference/config/report and original sources are untouched.
    for directory in [OUTPUT / "assets/visual", OUTPUT / "assets/collision"]:
        for path in directory.iterdir():
            if str(path.relative_to(OUTPUT)) not in manifest["assets"]:
                path.unlink()
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        OUTPUT / "scene.xml", encoding="utf-8", xml_declaration=True
    )
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUTPUT / "inertial_parameters.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "body",
                "mass_kg",
                "com_x_m",
                "com_y_m",
                "com_z_m",
                "Ixx_kg_m2",
                "Iyy_kg_m2",
                "Izz_kg_m2",
                "Ixy_kg_m2",
                "Ixz_kg_m2",
                "Iyz_kg_m2",
                "mass_class",
                "com_class",
                "inertia_class",
                "method",
            ]
        )
        for name, record in manifest["inertials"].items():
            writer.writerow(
                [
                    name,
                    record["mass_kg"],
                    *record["com_m"],
                    *tensor6(np.array(record["tensor_kg_m2"])),
                    record["mass_class"],
                    record["com_class"],
                    record["inertia_class"],
                    record["method"],
                ]
            )
    for name, record in manifest["inertials"].items():
        print(f"{name}: {record['mass_kg']:.9g} kg; {record['method']}")
    print("Mass budget:", manifest["mass_summary"])
    print("Direct mount:", manifest["mount"]["xyz"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download missing checksum-pinned official build inputs",
    )
    build(parser.parse_args().fetch)
