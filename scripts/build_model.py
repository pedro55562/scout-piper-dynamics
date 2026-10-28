#!/usr/bin/env python3
"""Offline, reproducible conversion of the supplied ROS assets; never needed to run MJCF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import collada
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ppscout_ws-dev/src"
OUTPUT = ROOT / "models/scout_piper"
SCOUT = SOURCE / "third_party/scout_ros2/scout_description"
PIPER = SOURCE / "third_party/agx_arm_urdf/piper/meshes"
DESCRIPTION = SOURCE / "ppscout_description/urdf"
MOUNT = np.array([0.0, 0.0, 0.054])
HOME_ARM = [0.0, 0.7, -1.4, 0.0, 0.7, 0.0]


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in np.asarray(values).ravel())


def vector(element, attribute, default="0 0 0"):
    return np.fromstring(element.get(attribute, default) if element is not None else default, sep=" ")


def pose(origin):
    result = {"pos": numbers(vector(origin, "xyz"))}
    rpy = vector(origin, "rpy")
    xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    result["quat"] = numbers(xyzw[[3, 0, 1, 2]])
    return result


def tensor(element):
    a = element.attrib
    return np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                     [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                     [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])


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
        out.write("# Converted from supplied Collada; units: meters. See source_manifest.json.\n")
        voffset = noffset = 1
        for vertices, faces, normals, normal_faces in primitives:
            np.savetxt(out, vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(out, normals, fmt="vn %.9g %.9g %.9g")
            for face, normal_face in zip(faces + voffset, normal_faces + noffset):
                out.write("f " + " ".join(f"{v}//{n}" for v, n in zip(face, normal_face)) + "\n")
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
            groups.setdefault(color, []).append((vertices, primitive.vertex_index, normals, primitive.normal_index))
            vertices_all.append(vertices)
            if name == "wheel" and np.allclose(color[:3], [0.2941176593] * 3):
                tire.append(vertices)
    visual_geoms = []
    for index, (color, primitives) in enumerate(groups.items()):
        mesh_name = f"{name}_visual_{index}"
        relative = f"assets/visual/{mesh_name}.obj"
        write_obj(OUTPUT / relative, primitives)
        ET.SubElement(asset, "mesh", name=mesh_name, file=relative, inertia="shell")
        visual_geoms.append({"name": mesh_name, "type": "mesh", "mesh": mesh_name,
                             "rgba": numbers(color), "class": "visual"})
        manifest["assets"][relative] = {"source": str(source_path.relative_to(ROOT)),
                                       "diffuse_rgba": color,
                                       "triangles": sum(len(p[1]) for p in primitives)}
    return visual_geoms, np.concatenate(vertices_all), np.concatenate(tire) if tire else None


def roof_heights(vertices, faces, xy):
    triangles = vertices[faces]
    a = triangles[:, 0, :2]
    u = triangles[:, 1, :2] - a
    v = triangles[:, 2, :2] - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    good = abs(det) > 1e-12
    r = xy - a
    b = np.divide(r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0], det, out=np.zeros_like(det), where=good)
    c = np.divide(u[:, 0] * r[:, 1] - u[:, 1] * r[:, 0], det, out=np.zeros_like(det), where=good)
    mask = good & (b >= -1e-8) & (c >= -1e-8) & (b + c <= 1 + 1e-8)
    z = triangles[:, 0, 2] + b * (triangles[:, 1, 2] - triangles[:, 0, 2]) + c * (triangles[:, 2, 2] - triangles[:, 0, 2])
    return float(z[mask].max())


def build():
    for subdir in ["assets/visual", "assets/collision", "licenses"]:
        (OUTPUT / subdir).mkdir(parents=True, exist_ok=True)
    scout_path = SCOUT / "urdf/model.xacro"
    arm_path = DESCRIPTION / "piper_description_scout.urdf"
    gripper_path = DESCRIPTION / "piper_with_gripper_scout.xacro"
    scout = ET.parse(scout_path).getroot()
    arm = ET.parse(arm_path).getroot()
    gripper = ET.parse(gripper_path).getroot()
    links = {l.get("name"): (l, path) for tree, path in [(scout, scout_path), (arm, arm_path), (gripper, gripper_path)] for l in tree.findall("link")}
    wheel_inertial = scout.find("{http://www.ros.org/wiki/xacro}macro[@name='wheel_inertial']/inertial")
    for name in ["front_left", "front_right", "rear_left", "rear_right"]:
        links[f"{name}_wheel_link"][0].append(ET.fromstring(ET.tostring(wheel_inertial)))
    manifest = {"description": "Deterministic conversion of local supplied files; no source files modified.",
                "source_files": {}, "assets": {}, "inertials": {}, "mount": {"xyz": MOUNT.tolist(), "rpy": [0, 0, 0]}}
    root = ET.Element("mujoco", model="Scout Mini with Piper")
    root.append(ET.Comment("Generated by scripts/build_model.py. SI units. See docs/model_notes.md."))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "statistic", center="0 0 0.45", extent="1.3")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="135", elevation="-20", offwidth="1440", offheight="1080")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction="1 0.005 0.0001")  # MuJoCo defaults, uncalibrated.
    vclass = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(vclass, "geom", contype="0", conaffinity="0", group="2", density="0")
    cclass = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(cclass, "geom", group="3", rgba="0.2 0.7 0.3 0.3", density="0")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key_light", pos="0 -2 3", dir="0 0.5 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="5 5 0.1", rgba="0.82 0.85 0.88 1")
    transforms = {"base_link": Rotation.from_euler("xyz", [1.5708, 0, -1.5708]).as_matrix()}
    mesh_files = {"base_link": SCOUT / "meshes/scout_mini_base_link2.dae", "wheel": SCOUT / "meshes/wheel.dae"}
    for name in ["piper_base_link", *[f"link{i}" for i in range(1, 7)], "flange_link", "gripper_base", "gripper_link1", "gripper_link2"]:
        filename = {"piper_base_link": "base_link", "flange_link": "flange"}.get(name, name)
        mesh_files[name] = PIPER / "dae" / f"{filename}.dae"
    converted = {}
    for name, source_path in mesh_files.items():
        vg, vertices, tire = visual_assets(name, source_path, transforms.get(name, np.eye(3)), asset, manifest)
        # Arm collision STLs are the supplied physical geometry; Scout has only Collada.
        physical_path = source_path if name in ("base_link", "wheel") else PIPER / (source_path.stem + ".stl")
        physical = vertices if physical_path == source_path else trimesh.load_mesh(physical_path).vertices
        hull = trimesh.convex.convex_hull(physical)
        if name not in ("base_link", "wheel"):
            relative = f"assets/collision/{name}_hull.stl"
            hull.export(OUTPUT / relative)
            ET.SubElement(asset, "mesh", name=f"{name}_hull", file=relative)
            manifest["assets"][relative] = {"source": str(physical_path.relative_to(ROOT)), "operation": "convex hull", "triangles": len(hull.faces)}
        converted[name] = (vg, vertices, tire, hull)
        manifest["source_files"][str(physical_path.relative_to(ROOT))] = hashlib.sha256(physical_path.read_bytes()).hexdigest()
        manifest["source_files"][str(source_path.relative_to(ROOT))] = hashlib.sha256(source_path.read_bytes()).hexdigest()

    # Contact cylinder follows the tire envelope, excluding the long decorative axle/hub.
    tire = converted["wheel"][2]
    tire_center = (tire.min(0) + tire.max(0)) / 2
    tire_radius = float(np.linalg.norm(tire[:, :2] - tire_center[:2], axis=1).max())
    tire_halfwidth = float(np.ptp(tire[:, 2]) / 2)
    manifest["wheel_collision"] = {"pos": tire_center.tolist(), "radius": tire_radius, "halfwidth": tire_halfwidth}
    # Raise the base just 1 mm above the lowest contact envelope at reset.
    wheel_joints = [j for j in scout.findall("joint") if j.get("name") in [f"{side}_wheel" for side in ["front_left", "front_right", "rear_left", "rear_right"]]]
    bottoms = []
    for j in wheel_joints:
        R = Rotation.from_euler("xyz", vector(j.find("origin"), "rpy")).as_matrix()
        center = vector(j.find("origin"), "xyz") + R @ tire_center
        axial_z = abs(R[2, 2])
        bottoms.append(center[2] - tire_radius * np.sqrt(1 - axial_z**2) - tire_halfwidth * axial_z)
    base_height = float(-min(bottoms) + 0.001)
    base = ET.SubElement(world, "body", name="base_link", pos=numbers([0, 0, base_height]))
    ET.SubElement(base, "freejoint", name="floating_base")
    bodies = {"base_link": base}

    def populate(name, body):
        key = "wheel" if name.endswith("_wheel_link") else name
        vg, vertices, _, hull = converted[key]
        for attributes in vg:
            attributes = dict(attributes)
            attributes["name"] = f"{name}_{attributes['name']}"
            ET.SubElement(body, "geom", attributes)
        if key == "wheel":
            ET.SubElement(body, "geom", {"name": f"{name}_contact", "class": "collision", "type": "cylinder",
                                        "pos": numbers(tire_center), "size": numbers([tire_radius, tire_halfwidth])})
        elif key == "base_link":
            lo, hi = vertices.min(0), vertices.max(0)
            hi[2] = MOUNT[2]  # Main deck, rather than side-rail height across the arm footprint.
            ET.SubElement(body, "geom", {"name": "chassis_contact", "class": "collision", "type": "box",
                                        "pos": numbers((lo + hi) / 2), "size": numbers((hi - lo) / 2)})
        else:
            ET.SubElement(body, "geom", {"name": f"{name}_contact", "class": "collision", "type": "mesh", "mesh": f"{name}_hull"})
        link, source_path = links[name]
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        center = vector(inertial.find("origin"), "xyz")
        R = Rotation.from_euler("xyz", vector(inertial.find("origin"), "rpy")).as_matrix()
        inertia = R @ tensor(inertial.find("inertia")) @ R.T
        bounds = hull.bounds
        reasons, variance, variance_limit = audit(mass, center, inertia, bounds)
        record = {"source": str(source_path.relative_to(ROOT)), "mass_kg": mass,
                  "source_com_m": center.tolist(), "source_tensor_kg_m2": inertia.tolist(),
                  "mesh_bounds_m": bounds.tolist(), "source_variance_m2": variance.tolist(),
                  "variance_bound_m2": variance_limit.tolist(), "rejection_reasons": reasons,
                  "method": "uniform-density convex hull at source mass" if reasons else "source unchanged (tensor rotated into link frame)"}
        if reasons:
            # No guessed scalar inertia: integrate the supplied geometry with its stated mass.
            center = hull.center_mass
            inertia = hull.moment_inertia * (mass / hull.mass)
            body.append(ET.Comment(" Source inertial rejected by mesh support audit; uniform hull at original mass. "))
        record.update({"com_m": center.tolist(), "tensor_kg_m2": inertia.tolist()})
        manifest["inertials"][name] = record
        ET.SubElement(body, "inertial", mass=numbers([mass]), pos=numbers(center), fullinertia=numbers(tensor6(inertia)))

    populate("base_link", base)

    def add_joint(j):
        name = j.get("name")
        child = j.find("child").get("link")
        parent = bodies[j.find("parent").get("link")]
        body = ET.SubElement(parent, "body", name=child, **pose(j.find("origin")))
        bodies[child] = body
        if j.get("type") != "fixed":
            attributes = {"name": name, "type": "slide" if j.get("type") == "prismatic" else "hinge", "axis": j.find("axis").get("xyz")}
            limit = j.find("limit")
            if limit is not None:
                attributes["range"] = f"{limit.get('lower')} {limit.get('upper')}"
            dynamics = j.find("dynamics")
            if dynamics is not None:
                attributes["damping"] = dynamics.get("damping", "0")
                attributes["frictionloss"] = dynamics.get("friction", "0")
            if name in ("gripper_joint1", "gripper_joint2"):
                # opening=2*q1=-2*q2: split the virtual slider's dissipated power equally.
                opening_dynamics = gripper.find("joint[@name='gripper']/dynamics")
                attributes["damping"] = numbers([2 * float(opening_dynamics.get("damping"))])
                attributes["frictionloss"] = opening_dynamics.get("friction")
            ET.SubElement(body, "joint", attributes)
        populate(child, body)

    for j in wheel_joints:
        add_joint(j)
    piper_base = ET.SubElement(base, "body", name="piper_base_link", pos=numbers(MOUNT))
    piper_base.append(ET.Comment(" Rigid direct deck mount: xyz=(0,0,0.054) m, rpy=(0,0,0). "))
    bodies["piper_base_link"] = piper_base
    populate("piper_base_link", piper_base)
    for j in arm.findall("joint"):
        add_joint(j)
    for j in gripper.findall("joint"):
        if j.get("name") != "gripper":
            add_joint(j)
    equality = ET.SubElement(root, "equality")
    equality.append(ET.Comment(" Source mimic: q1=opening/2, q2=-opening/2. Remove nonphysical dummy opening slider. "))
    ET.SubElement(equality, "joint", name="gripper_symmetry", joint1="gripper_joint2", joint2="gripper_joint1", polycoef="0 -1 0 0 0", solref="0.002 1", solimp="0.99 0.999 0.001")
    keyframe = ET.SubElement(root, "keyframe")
    qpos = [0, 0, base_height, 1, 0, 0, 0, *([0] * 4), *HOME_ARM, 0.02, -0.02]
    ET.SubElement(keyframe, "key", name="home", qpos=numbers(qpos))

    # Check deck across the complete rectangular Piper base footprint using vertical rays.
    d = collada.Collada(str(mesh_files["base_link"]))
    vertices, faces, offset = [], [], 0
    for g in d.scene.objects("geometry"):
        for p in g.primitives():
            if hasattr(p, "triangleset"):
                p = p.triangleset()
            vertices.append(np.asarray(p.vertex) @ transforms["base_link"].T)
            faces.append(p.vertex_index + offset)
            offset += len(p.vertex)
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    lo, hi = converted["piper_base_link"][1].min(0), converted["piper_base_link"][1].max(0)
    heights = [roof_heights(vertices, faces, [x, y]) for x in np.linspace(lo[0], hi[0], 11) for y in np.linspace(lo[1], hi[1], 11)]
    manifest["mount"].update({"piper_base_bounds_m": [lo.tolist(), hi.tolist()], "deck_grid": [11, 11], "deck_z_range_m": [min(heights), max(heights)], "min_clearance_m": MOUNT[2] + lo[2] - max(heights)})
    triangles = vertices[faces]
    overlaps = np.all(triangles[:, :, :2].max(1) >= lo[:2], axis=1) & np.all(triangles[:, :, :2].min(1) <= hi[:2], axis=1)
    manifest["mount"]["conservative_deck_z_max_m"] = float(triangles[overlaps, :, 2].max())
    assert manifest["mount"]["conservative_deck_z_max_m"] <= MOUNT[2] + lo[2]
    assert 0 <= manifest["mount"]["min_clearance_m"] < 1e-5
    manifest["initial_base_height_m"] = base_height
    manifest["home_arm_rad"] = HOME_ARM
    manifest["gripper_reduction"] = {"opening_m": "2*q1 = -2*q2", "removed_virtual_mass_kg": 0.001,
                                      "finger_damping_Ns_per_m": 1.0, "finger_friction_N": 0.05,
                                      "method": "equal power-preserving split of source virtual opening joint damping 0.5 and friction 0.05"}
    for path in [scout_path, arm_path, gripper_path, DESCRIPTION / "scout_piper.urdf.xacro"]:
        manifest["source_files"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for src, dest in [(SOURCE / "third_party/agx_arm_urdf/LICENSE", "Piper-MIT.txt"), (SOURCE / "third_party/scout_ros2/LICENSE", "Scout-Apache-2.0.txt")]:
        shutil.copyfile(src, OUTPUT / "licenses" / dest)
        manifest["source_files"][str(src.relative_to(ROOT))] = hashlib.sha256(src.read_bytes()).hexdigest()
    for relative, record in manifest["assets"].items():
        record["sha256"] = hashlib.sha256((OUTPUT / relative).read_bytes()).hexdigest()
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUTPUT / "scene.xml", encoding="utf-8", xml_declaration=True)
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, record in manifest["inertials"].items():
        print(f"{name}: {record['mass_kg']:g} kg; {record['method']}")
    print(f"Wrote {OUTPUT / 'scene.xml'}; base reset height {base_height:.9f} m")


if __name__ == "__main__":
    build()
