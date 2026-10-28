#!/usr/bin/env python3
"""Repeatable structural, physical, contact and optional source/visual checks."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import unittest
import xml.etree.ElementTree as ET
import zlib

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/scout_piper"
MODEL_PATH = MODEL_DIR / "scene.xml"
MANIFEST = json.loads((MODEL_DIR / "source_manifest.json").read_text())
WHEELS = [f"{side}_wheel" for side in ["front_left", "front_right", "rear_left", "rear_right"]]
ARM = [f"joint{i}" for i in range(1, 7)]
FINGERS = ["gripper_joint1", "gripper_joint2"]


def load():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def urdf_transform(origin):
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    r, p, y = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    result = np.eye(4)
    result[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    result[:3, 3] = xyz
    return result


def compiled_mesh(model, data, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
    vertices = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    start = model.mesh_faceadr[mesh_id]
    faces = model.mesh_face[start:start + model.mesh_facenum[mesh_id]]
    return vertices, faces


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.model, self.data = load()

    def test_structure(self):
        m, d = self.model, self.data
        self.assertEqual({m.joint(i).name for i in range(m.njnt)}, {"floating_base", *WHEELS, *ARM, *FINGERS})
        self.assertEqual({m.body(i).name for i in range(1, m.nbody)},
                         {"base_link", "piper_base_link", *[f"{n}_link" for n in WHEELS],
                          *[f"link{i}" for i in range(1, 7)], "flange_link", "gripper_base", "gripper_link1", "gripper_link2"})
        self.assertEqual((m.nbody, m.njnt, m.nq, m.nv, m.neq, m.nu, m.nsensor), (17, 13, 19, 18, 1, 0, 0))
        self.assertEqual(m.joint("floating_base").type, mujoco.mjtJoint.mjJNT_FREE)
        for name in WHEELS + ARM:
            self.assertEqual(m.joint(name).type, mujoco.mjtJoint.mjJNT_HINGE)
        for name in WHEELS:
            self.assertFalse(m.joint(name).limited)
        for name in FINGERS:
            self.assertEqual(m.joint(name).type, mujoco.mjtJoint.mjJNT_SLIDE)
        np.testing.assert_array_equal(m.opt.gravity, [0, 0, -9.81])
        self.assertEqual(d.ncon, 0, "Home must begin without overlapping collision geometry")

    def test_standalone_assets_and_source_integrity(self):
        xml = ET.parse(MODEL_PATH).getroot()
        files = {e.get("file") for e in xml.findall("asset/mesh")}
        self.assertEqual(files, set(MANIFEST["assets"]))
        for relative in files:
            path = MODEL_DIR / relative
            self.assertTrue(path.is_file())
            self.assertTrue(path.resolve().is_relative_to(MODEL_DIR.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), MANIFEST["assets"][relative]["sha256"])
        self.assertFalse(xml.findall("include"))
        # Optional audit: the model and all remaining tests work without the source tree.
        for relative, digest in MANIFEST["source_files"].items():
            path = ROOT / relative
            if path.exists():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, relative)

    def test_direct_mount_geometry(self):
        m, d = self.model, self.data
        base = m.body("base_link").id
        piper = m.body("piper_base_link").id
        self.assertEqual(m.body_parentid[piper], base)
        self.assertEqual(m.body_jntnum[piper], 0)
        np.testing.assert_allclose(m.body_pos[piper], [0, 0, 0.054], atol=1e-12)
        np.testing.assert_allclose(m.body_quat[piper], [1, 0, 0, 0], atol=1e-12)
        # Check the actual compiled visual meshes, accounting for MuJoCo's mesh recentering.
        def mesh_in_base(g):
            v, f = compiled_mesh(m, d, g)
            return (v - d.xpos[base]) @ d.xmat[base].reshape(3, 3), f
        piper_vertices = np.concatenate([mesh_in_base(g)[0] for g in range(m.ngeom)
                                         if m.geom_bodyid[g] == piper and m.geom_group[g] == 2])
        bottom = piper_vertices[:, 2].min()
        lo, hi = piper_vertices[:, :2].min(0), piper_vertices[:, :2].max(0)
        triangles = np.concatenate([v[f] for g in range(m.ngeom)
                                    if m.geom_bodyid[g] == base and m.geom_group[g] == 2
                                    for v, f in [mesh_in_base(g)]])
        overlaps = np.all(triangles[:, :, :2].max(1) >= lo, axis=1) & np.all(triangles[:, :, :2].min(1) <= hi, axis=1)
        self.assertLessEqual(triangles[overlaps, :, 2].max(), bottom + 1e-7,
                             "Conservative triangle bound must clear the complete Piper footprint")
        a = triangles[:, 0, :2]
        u = triangles[:, 1, :2] - a
        v = triangles[:, 2, :2] - a
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        valid = abs(det) > 1e-12
        roof = []
        for x in np.linspace(lo[0], hi[0], 11):
            for y in np.linspace(lo[1], hi[1], 11):
                r = [x, y] - a
                b = np.divide(r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0], det, out=np.zeros_like(det), where=valid)
                c = np.divide(u[:, 0] * r[:, 1] - u[:, 1] * r[:, 0], det, out=np.zeros_like(det), where=valid)
                mask = valid & (b >= -1e-7) & (c >= -1e-7) & (b + c <= 1 + 1e-7)
                z = triangles[:, 0, 2] + b * (triangles[:, 1, 2] - triangles[:, 0, 2]) + c * (triangles[:, 2, 2] - triangles[:, 0, 2])
                self.assertTrue(mask.any())
                roof.append(z[mask].max())
        clearance = bottom - max(roof)
        self.assertGreaterEqual(clearance, -1e-7)
        self.assertLess(clearance, 1e-5, "Arm should sit directly on the deck")

    def test_inertial_parameters(self):
        m = self.model
        for name, record in MANIFEST["inertials"].items():
            body = m.body(name)
            self.assertAlmostEqual(float(body.mass[0]), record["mass_kg"], places=10)
            np.testing.assert_allclose(body.ipos, record["com_m"], atol=1e-11)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, body.iquat)
            R = R.reshape(3, 3)
            inertia = R @ np.diag(body.inertia) @ R.T
            np.testing.assert_allclose(inertia, record["tensor_kg_m2"], atol=1e-10)
            self.assertTrue(np.all(body.inertia > 0))
            eigen = np.sort(body.inertia)
            self.assertLessEqual(eigen[-1], eigen[:2].sum() + 1e-10)
            lo, hi = np.asarray(record["mesh_bounds_m"])
            center = body.ipos
            variance = (np.trace(inertia) / 2 - np.diag(inertia)) / record["mass_kg"]
            self.assertTrue(np.all(variance <= (center - lo) * (hi - center) + 1e-9), name)
            if not record["rejection_reasons"]:
                np.testing.assert_allclose(inertia, record["source_tensor_kg_m2"], atol=1e-10)
        self.assertAlmostEqual(float(m.body_mass.sum()), 76.7, places=8)

    @unittest.skipUnless((ROOT / "ppscout_ws-dev/src").is_dir(), "Optional source tree is absent")
    def test_source_kinematics(self):
        source = ROOT / "ppscout_ws-dev/src"
        scout = ET.parse(source / "third_party/scout_ros2/scout_description/urdf/model.xacro").getroot()
        arm = ET.parse(source / "ppscout_description/urdf/piper_description_scout.urdf").getroot()
        gripper = ET.parse(source / "ppscout_description/urdf/piper_with_gripper_scout.xacro").getroot()
        joints = [j for j in scout.findall("joint") if j.get("name") in WHEELS]
        mount = ET.fromstring('<joint name="mount" type="fixed"><parent link="base_link"/><child link="piper_base_link"/><origin xyz="0 0 0.054" rpy="0 0 0"/></joint>')
        joints += [mount, *arm.findall("joint"), *[j for j in gripper.findall("joint") if j.get("name") != "gripper"]]
        m, d = self.model, self.data
        rng = np.random.default_rng(42)
        for _ in range(20):
            d.qpos[:3] = rng.uniform(-0.3, 0.3, 3)
            d.qpos[3:7] = rng.normal(size=4)
            d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
            for name in WHEELS + ARM + FINGERS:
                joint = m.joint(name)
                low, high = joint.range if joint.limited else [-np.pi, np.pi]
                d.joint(name).qpos[0] = rng.uniform(low, high)
            mujoco.mj_forward(m, d)
            base = np.eye(4)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            base[:3, :3], base[:3, 3] = R.reshape(3, 3), d.qpos[:3]
            transforms = {"base_link": base}
            for joint in joints:
                T = transforms[joint.find("parent").get("link")] @ urdf_transform(joint.find("origin"))
                if joint.get("type") != "fixed":
                    q = d.joint(joint.get("name")).qpos[0]
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    movement = np.eye(4)
                    if joint.get("type") == "prismatic":
                        movement[:3, 3] = axis * q
                    else:
                        movement[:3, :3] = rotation(axis, q)
                    T = T @ movement
                child = joint.find("child").get("link")
                transforms[child] = T
                np.testing.assert_allclose(d.body(child).xpos, T[:3, 3], atol=1e-9, err_msg=child)
                np.testing.assert_allclose(d.body(child).xmat.reshape(3, 3), T[:3, :3], atol=1e-9, err_msg=child)
                if joint.get("type") != "fixed":
                    source_limit = joint.find("limit")
                    if source_limit is not None:
                        np.testing.assert_allclose(m.joint(joint.get("name")).range,
                                                   [float(source_limit.get("lower")), float(source_limit.get("upper"))], atol=1e-12)

    def test_gripper_opening(self):
        m, d = self.model, self.data
        origin = d.body("gripper_base").xpos.copy()
        R = d.body("gripper_base").xmat.reshape(3, 3).copy()
        for opening in [0, 0.02, 0.05, 0.1]:
            d.joint("gripper_joint1").qpos[0] = opening / 2
            d.joint("gripper_joint2").qpos[0] = -opening / 2
            mujoco.mj_forward(m, d)
            first = R.T @ (d.body("gripper_link1").xpos - origin)
            second = R.T @ (d.body("gripper_link2").xpos - origin)
            self.assertAlmostEqual(second[1] - first[1], opening, places=7)
            self.assertLess(abs(d.efc_pos[0]), 1e-9)

    def test_passive_simulation_and_wheel_contacts(self):
        m, d = self.model, self.data
        wheels_seen = set()
        max_penetration = max_gripper_error = 0.0
        for _ in range(round(10 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            self.assertTrue(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all())
            max_gripper_error = max(max_gripper_error, abs(d.joint(FINGERS[0]).qpos[0] + d.joint(FINGERS[1]).qpos[0]))
            for c in d.contact:
                names = {m.geom(c.geom1).name, m.geom(c.geom2).name}
                if "ground" in names:
                    wheels_seen |= names - {"ground"}
                max_penetration = max(max_penetration, -c.dist)
        self.assertEqual(wheels_seen, {f"{w}_link_contact" for w in WHEELS})
        self.assertTrue(all(w.number == 0 for w in d.warning))
        self.assertAlmostEqual(d.time, 10.0, places=8)
        self.assertGreater(d.qpos[2], 0.17)
        self.assertLess(max_penetration, 0.01)
        self.assertLess(max_gripper_error, 1e-4)
        supporting = set()
        for i, c in enumerate(d.contact):
            names = {m.geom(c.geom1).name, m.geom(c.geom2).name}
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if "ground" in names and force[0] > 0:
                supporting |= names - {"ground"}
        self.assertEqual(supporting, {f"{w}_link_contact" for w in WHEELS})
        print(f"\n  10 s passive run: all four wheels support the robot; "
              f"max penetration={max_penetration * 1000:.3f} mm; "
              f"max finger symmetry error={max_gripper_error * 1000:.4f} mm; no warnings.")


def render(path):
    m, d = load()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.43]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -20
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(m, height=1080, width=1440) as renderer:
        renderer.update_scene(d, camera=camera, scene_option=option)
        pixels = renderer.render()
    # Write RGB PNG with the standard library, without an extra image dependency.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    height, width, _ = pixels.shape
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
    print(f"Rendered {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, help="Also save a PNG; requires OpenGL (MUJOCO_GL=egl on headless Linux)")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.render:
        render(args.render)
