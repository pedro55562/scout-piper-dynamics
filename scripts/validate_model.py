#!/usr/bin/env python3
"""Structural, official-reference FK, inertial, torque and passive-contact checks."""

import argparse
import hashlib
import json
import platform
import struct
import unittest
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import mujoco
import numpy as np
from arm_dynamics import configure_arm_dynamics, configure_wheel_motors

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/scout_piper"
MODEL_PATH = MODEL_DIR / "scene.xml"
MANIFEST = json.loads((MODEL_DIR / "source_manifest.json").read_text())
WHEELS = [
    f"{side}_wheel" for side in ["front_left", "front_right", "rear_left", "rear_right"]
]
ARM = [f"joint{i}" for i in range(1, 7)]
FINGERS = ["gripper_joint1", "gripper_joint2"]
METRICS = {}


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
    result[:3, :3] = (
        rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    )
    result[:3, 3] = xyz
    return result


def compiled_mesh(model, data, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
    vertices = (
        vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    )
    start = model.mesh_faceadr[mesh_id]
    faces = model.mesh_face[start : start + model.mesh_facenum[mesh_id]]
    return vertices, faces


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.model, self.data = load()

    def test_structure(self):
        m, d = self.model, self.data
        self.assertEqual(
            {m.joint(i).name for i in range(m.njnt)},
            {"floating_base", *WHEELS, *ARM, *FINGERS},
        )
        self.assertEqual(
            {m.body(i).name for i in range(1, m.nbody)},
            {
                "base_link",
                "piper_base_link",
                *[f"{n}_link" for n in WHEELS],
                *[f"link{i}" for i in range(1, 7)],
                "flange_link",
                "gripper_base",
                "gripper_link1",
                "gripper_link2",
            },
        )
        self.assertEqual(
            (m.nbody, m.njnt, m.nq, m.nv, m.neq, m.nu, m.nsensor),
            (17, 13, 19, 18, 1, 10, 0),
        )
        self.assertEqual(m.joint("floating_base").type, mujoco.mjtJoint.mjJNT_FREE)
        for name in WHEELS + ARM:
            self.assertEqual(m.joint(name).type, mujoco.mjtJoint.mjJNT_HINGE)
        for name in WHEELS:
            self.assertFalse(m.joint(name).limited)
        for name in FINGERS:
            self.assertEqual(m.joint(name).type, mujoco.mjtJoint.mjJNT_SLIDE)
        np.testing.assert_array_equal(m.opt.gravity, [0, 0, -9.81])
        self.assertEqual(
            d.ncon, 0, "Home must begin without overlapping collision geometry"
        )

    def test_direct_torque_actuators(self):
        m, d = self.model, self.data
        xml = ET.parse(MODEL_PATH).getroot()
        self.assertEqual([a.tag for a in xml.find("actuator")], ["motor"] * 10)
        ET.indent(xml, space="  ")
        before = ET.tostring(xml)
        configure_arm_dynamics(
            xml,
            {name: MANIFEST["arm_dynamics"]["joints"][name]["source"] for name in ARM},
        )
        configure_wheel_motors(
            xml, MANIFEST["assumptions"]["scout"]["torque_limits_Nm"]
        )
        ET.indent(xml, space="  ")
        self.assertEqual(
            ET.tostring(xml),
            before,
            "Rebuild configuration must match the delivered scene",
        )
        self.assertEqual(m.na, 0, "Torque motors must have no activation dynamics")
        np.testing.assert_array_equal(m.actuator_biasprm, 0)
        for name in ARM + WHEELS:
            actuator = m.actuator(f"{name}_motor")
            joint = m.joint(name)
            dof = joint.dofadr[0]
            self.assertEqual(actuator.trntype[0], mujoco.mjtTrn.mjTRN_JOINT)
            self.assertEqual(actuator.trnid[0], joint.id)
            self.assertEqual(actuator.gaintype[0], mujoco.mjtGain.mjGAIN_FIXED)
            self.assertEqual(actuator.biastype[0], mujoco.mjtBias.mjBIAS_NONE)
            self.assertEqual(actuator.dyntype[0], mujoco.mjtDyn.mjDYN_NONE)
            np.testing.assert_array_equal(actuator.gear, [1, 0, 0, 0, 0, 0])
            np.testing.assert_array_equal(
                actuator.gainprm, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
            self.assertTrue(actuator.ctrllimited[0])
            limit = (
                MANIFEST["arm_dynamics"]["joints"][name]["source"]["command_limit_Nm"]
                if name in ARM
                else MANIFEST["wheel_dynamics"]["joints"][name]["ctrlrange_Nm"][1]
            )
            np.testing.assert_array_equal(actuator.ctrlrange, [-limit, limit])
            # Verify sign, units, isolation and saturation using generalized force, not just XML.
            for command in [-2 * limit, -1.75, 1.75, 2 * limit]:
                d.ctrl[:] = 0
                d.ctrl[actuator.id] = command
                mujoco.mj_forward(m, d)
                expected = np.zeros(m.nv)
                expected[dof] = np.clip(command, -limit, limit)
                np.testing.assert_allclose(d.qfrc_actuator, expected, atol=1e-12)

    def test_no_arm_pose_holding_forces(self):
        m, d = self.model, self.data
        np.testing.assert_array_equal(m.body_gravcomp, 0)
        np.testing.assert_array_equal(m.jnt_stiffness, 0)
        np.testing.assert_array_equal(m.dof_armature, 0)
        self.assertEqual(m.ntendon, 0)
        self.assertEqual(m.opt.viscosity, 0)
        self.assertEqual(m.opt.density, 0)
        self.assertFalse(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        self.assertFalse(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        arm_dofs = [m.joint(name).dofadr[0] for name in ARM]
        np.testing.assert_array_equal(m.dof_damping[arm_dofs], 0)
        np.testing.assert_array_equal(m.dof_frictionloss[arm_dofs], 0)
        d.qvel[arm_dofs] = [0.2, -0.4, 0.6, -0.8, 1, -1.2]
        mujoco.mj_forward(m, d)
        np.testing.assert_array_equal(d.qfrc_passive[arm_dofs], 0)
        for name in WHEELS:
            self.assertEqual(m.dof_damping[m.joint(name).dofadr[0]], 0)
            self.assertEqual(m.dof_frictionloss[m.joint(name).dofadr[0]], 0)
        np.testing.assert_array_equal(m.dof_damping, 0)
        np.testing.assert_array_equal(m.dof_frictionloss, 0)

    def test_gravity_drives_zero_command_motion(self):
        m, d = self.model, self.data
        dofs = [m.joint(name).qposadr[0] for name in ARM]
        initial = d.qpos[dofs].copy()
        for _ in range(round(0.25 / m.opt.timestep)):
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)
            np.testing.assert_array_equal(d.actuator_force, 0)
            np.testing.assert_array_equal(d.qfrc_actuator, 0)
            np.testing.assert_array_equal(d.qfrc_applied, 0)
            np.testing.assert_array_equal(d.xfrc_applied, 0)
        self.assertGreater(np.max(abs(d.qpos[dofs] - initial)), 0.01)
        METRICS["gravity_response_0_25_s"] = {
            "max_joint_change_rad": float(np.max(abs(d.qpos[dofs] - initial))),
            "actuator_torque_Nm": 0.0,
        }
        # Diagnostic reference only: with gravity disabled the same zero-input pose stays still.
        reference, reference_data = load()
        reference.opt.gravity[:] = 0
        for _ in range(round(0.25 / reference.opt.timestep)):
            mujoco.mj_step(reference, reference_data)
        np.testing.assert_allclose(reference_data.qpos[dofs], initial, atol=1e-12)

    def test_standalone_assets_and_source_integrity(self):
        xml = ET.parse(MODEL_PATH).getroot()
        files = {e.get("file") for e in xml.findall("asset/mesh")}
        self.assertEqual(files, set(MANIFEST["assets"]))
        for relative in files:
            path = MODEL_DIR / relative
            self.assertTrue(path.is_file())
            self.assertTrue(path.resolve().is_relative_to(MODEL_DIR.resolve()))
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                MANIFEST["assets"][relative]["sha256"],
            )
        self.assertFalse(xml.findall("include"))
        for relative, digest in MANIFEST["reference_files"].items():
            self.assertEqual(
                hashlib.sha256((MODEL_DIR / relative).read_bytes()).hexdigest(), digest
            )
        # Optional audit: the model and all remaining tests work without the source tree.
        for relative, digest in MANIFEST["source_files"].items():
            path = ROOT / relative
            if path.exists():
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), digest, relative
                )

    def test_direct_mount_geometry(self):
        m, d = self.model, self.data
        base = m.body("base_link").id
        piper = m.body("piper_base_link").id
        self.assertEqual(m.body_parentid[piper], base)
        self.assertEqual(m.body_jntnum[piper], 0)
        np.testing.assert_allclose(
            m.body_pos[piper], MANIFEST["mount"]["xyz"], atol=1e-12
        )
        np.testing.assert_allclose(m.body_quat[piper], [1, 0, 0, 0], atol=1e-12)

        # Check the actual compiled visual meshes, accounting for MuJoCo's mesh recentering.
        def mesh_in_base(g):
            v, f = compiled_mesh(m, d, g)
            return (v - d.xpos[base]) @ d.xmat[base].reshape(3, 3), f

        piper_vertices = np.concatenate(
            [
                mesh_in_base(g)[0]
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == piper and m.geom_group[g] == 2
            ]
        )
        bottom = piper_vertices[:, 2].min()
        lo, hi = piper_vertices[:, :2].min(0), piper_vertices[:, :2].max(0)
        triangles = np.concatenate(
            [
                v[f]
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == base and m.geom_group[g] == 2
                for v, f in [mesh_in_base(g)]
            ]
        )
        overlaps = np.all(triangles[:, :, :2].max(1) >= lo, axis=1) & np.all(
            triangles[:, :, :2].min(1) <= hi, axis=1
        )
        self.assertLessEqual(
            triangles[overlaps, :, 2].max(),
            bottom + 1e-7,
            "Conservative triangle bound must clear the complete Piper footprint",
        )
        a = triangles[:, 0, :2]
        u = triangles[:, 1, :2] - a
        v = triangles[:, 2, :2] - a
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        valid = abs(det) > 1e-12
        roof = []
        for x in np.linspace(lo[0], hi[0], 11):
            for y in np.linspace(lo[1], hi[1], 11):
                r = [x, y] - a
                b = np.divide(
                    r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0],
                    det,
                    out=np.zeros_like(det),
                    where=valid,
                )
                c = np.divide(
                    u[:, 0] * r[:, 1] - u[:, 1] * r[:, 0],
                    det,
                    out=np.zeros_like(det),
                    where=valid,
                )
                mask = valid & (b >= -1e-7) & (c >= -1e-7) & (b + c <= 1 + 1e-7)
                z = (
                    triangles[:, 0, 2]
                    + b * (triangles[:, 1, 2] - triangles[:, 0, 2])
                    + c * (triangles[:, 2, 2] - triangles[:, 0, 2])
                )
                self.assertTrue(mask.any())
                roof.append(z[mask].max())
        clearance = bottom - max(roof)
        self.assertGreaterEqual(clearance, -1e-7)
        self.assertLess(
            clearance,
            2e-5,
            "Arm should sit directly on the deck, allowing source rotation rounding",
        )

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
            self.assertTrue(
                np.all(center >= lo - 1e-9) and np.all(center <= hi + 1e-9), name
            )
            variance = (np.trace(inertia) / 2 - np.diag(inertia)) / record["mass_kg"]
            self.assertTrue(
                np.all(variance <= (center - lo) * (hi - center) + 1e-9), name
            )
            if not record["rejection_reasons"]:
                np.testing.assert_allclose(
                    inertia, record["source_tensor_kg_m2"], atol=1e-10
                )
        self.assertAlmostEqual(float(m.body_mass.sum()), 27.70955408133, places=9)
        self.assertEqual(MANIFEST["mass_summary"]["scout_kg"], 23)
        self.assertEqual(
            MANIFEST["mass_summary"]["chassis_kg"]
            + MANIFEST["mass_summary"]["wheels_kg"],
            23,
        )
        self.assertAlmostEqual(MANIFEST["mass_summary"]["gripper_kg"], 0.5, places=10)
        self.assertAlmostEqual(
            MANIFEST["mass_summary"]["arm_with_flange_kg"], 4.20955408133, places=10
        )

    def test_source_kinematics(self):
        reference = MODEL_DIR / "reference"
        normalized = ET.parse(reference / "normalized_kinematics.urdf").getroot()
        joints = normalized.findall("joint")
        # Arm/finger poses and limits must also agree with the unchanged official snapshots.
        for filename in [
            "piper_description.urdf",
            "piper_with_gripper_description.xacro",
        ]:
            for j in ET.parse(reference / filename).getroot().findall("joint"):
                if j.get("name") in ("gripper", "world_to_base_link"):
                    continue
                selected = normalized.find(f"joint[@name='{j.get('name')}']")
                for field in ["origin", "axis", "limit"]:
                    a, b = j.find(field), selected.find(field)
                    self.assertEqual(
                        None if a is None else a.attrib, None if b is None else b.attrib
                    )
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
                T = transforms[joint.find("parent").get("link")] @ urdf_transform(
                    joint.find("origin")
                )
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
                np.testing.assert_allclose(
                    d.body(child).xpos, T[:3, 3], atol=1e-9, err_msg=child
                )
                np.testing.assert_allclose(
                    d.body(child).xmat.reshape(3, 3),
                    T[:3, :3],
                    atol=1e-9,
                    err_msg=child,
                )
                if joint.get("type") != "fixed":
                    source_limit = joint.find("limit")
                    if source_limit is not None:
                        np.testing.assert_allclose(
                            m.joint(joint.get("name")).range,
                            [
                                float(source_limit.get("lower")),
                                float(source_limit.get("upper")),
                            ],
                            atol=1e-12,
                        )

    def test_wheel_placement_and_direction(self):
        m, d = self.model, self.data
        raw = ET.parse(MODEL_DIR / "reference/scout_mini.urdf.xacro").getroot()
        props = {
            e.get("name"): float(e.get("value"))
            for e in raw.findall("{http://ros.org/wiki/xacro}property")
        }
        for name in WHEELS:
            body = m.body(name + "_link")
            left = "left" in name
            x = 0.451 / 2 * (1 if "front" in name else -1)
            offset = MANIFEST["wheel_collision"]["pos"][2]
            y = (0.490 / 2 - offset) * (1 if left else -1)
            np.testing.assert_allclose(
                body.pos, [x, y, props["wheel_vertical_offset"]], atol=1e-12
            )
            axis = d.body(name + "_link").xmat.reshape(3, 3) @ m.joint(name).axis
            np.testing.assert_allclose(axis, [0, 1, 0], atol=1e-10)
            self.assertLess(
                np.cross(axis, [0, 0, -1])[0],
                0,
                "Positive spin must roll the base forward",
            )
            cylinder = m.geom(name + "_link_contact")
            np.testing.assert_allclose(cylinder.pos[:2], [0, 0], atol=1e-12)
            # The original radial mesh export offset must not become rotating eccentricity.
            geom = m.geom(name + "_link_wheel_visual_2").id
            vertices, _ = compiled_mesh(m, d, geom)
            vertices = (vertices - d.body(name + "_link").xpos) @ d.body(
                name + "_link"
            ).xmat.reshape(3, 3)
            np.testing.assert_allclose(
                (vertices[:, :2].min(0) + vertices[:, :2].max(0)) / 2, [0, 0], atol=2e-8
            )
        self.assertAlmostEqual(props["wheelbase"], 0.463951, places=10)
        # Prefer manufacturer drawing dimensions; do not scale mesh geometry.
        self.assertAlmostEqual(
            2 * m.body("front_left_wheel_link").pos[0], 0.451, places=10
        )
        left = d.geom("front_left_wheel_link_contact").xpos
        right = d.geom("front_right_wheel_link_contact").xpos
        self.assertAlmostEqual(left[1] - right[1], 0.490, places=10)

    def test_official_inertial_provenance(self):
        for filename in [
            "piper_description.urdf",
            "piper_with_gripper_description.xacro",
        ]:
            for link in (
                ET.parse(MODEL_DIR / "reference" / filename).getroot().findall("link")
            ):
                ine = link.find("inertial")
                if ine is None:
                    continue
                name = (
                    "piper_base_link"
                    if link.get("name") == "base_link"
                    else link.get("name")
                )
                record = MANIFEST["inertials"][name]
                self.assertEqual(
                    record["mass_kg"], float(ine.find("mass").get("value"))
                )
                np.testing.assert_allclose(
                    record["source_com_m"],
                    np.fromstring(ine.find("origin").get("xyz"), sep=" "),
                )
                a = ine.find("inertia").attrib
                expected = [
                    [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                    [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                    [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
                ]
                np.testing.assert_allclose(
                    record["source_tensor_kg_m2"], expected, atol=1e-14
                )
                if record["inertia_class"] == 2:
                    np.testing.assert_allclose(
                        record["tensor_kg_m2"], expected, atol=1e-14
                    )
        self.assertEqual(MANIFEST["inertials"]["base_link"]["mass_class"], 4)
        self.assertEqual(MANIFEST["inertials"]["link6"]["mass_class"], 1)
        self.assertEqual(MANIFEST["inertials"]["link6"]["inertia_class"], 3)

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

    def test_manual_zero_pose(self):
        m, d = self.model, self.data
        for name in ARM:
            d.joint(name).qpos[0] = 0
        mujoco.mj_forward(m, d)
        base = d.body("piper_base_link")
        tip = d.body("link6")
        R = base.xmat.reshape(3, 3).T
        # Standard Piper manual, V1.0 2025.06, section 2.2.4: J6 frame at zero.
        # URDF's 88.259 µm J6 x offset produces an approximately 0.088 mm discrepancy.
        np.testing.assert_allclose(
            R @ (tip.xpos - base.xpos), [0.056128, 0, 0.213266], atol=1e-4
        )
        np.testing.assert_allclose(
            R @ tip.xmat.reshape(3, 3), rotation([0, 1, 0], np.deg2rad(85)), atol=2e-6
        )

    def test_passive_simulation_and_wheel_contacts(self):
        m, d = self.model, self.data
        wheels_seen = set()
        max_penetration = max_gripper_error = 0.0
        max_ground_penetration = 0.0
        for _ in range(round(10 / m.opt.timestep)):
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)
            self.assertFalse(
                np.any(d.ctrl) or np.any(d.actuator_force) or np.any(d.qfrc_actuator)
            )
            self.assertTrue(
                np.isfinite(d.qpos).all()
                and np.isfinite(d.qvel).all()
                and np.isfinite(d.qacc).all()
            )
            max_gripper_error = max(
                max_gripper_error,
                abs(d.joint(FINGERS[0]).qpos[0] + d.joint(FINGERS[1]).qpos[0]),
            )
            for c in d.contact:
                names = {m.geom(c.geom1).name, m.geom(c.geom2).name}
                if "ground" in names:
                    wheels_seen |= names - {"ground"}
                    max_ground_penetration = max(max_ground_penetration, -c.dist)
                max_penetration = max(max_penetration, -c.dist)
        self.assertEqual(wheels_seen, {f"{w}_link_contact" for w in WHEELS})
        self.assertTrue(all(w.number == 0 for w in d.warning))
        self.assertAlmostEqual(d.time, 10.0, places=8)
        self.assertGreater(d.qpos[2], 0.17)
        # Ground support remains bounded; undamped arm impacts are reported separately.
        self.assertLess(max_ground_penetration, 0.005)
        self.assertLess(max_gripper_error, 1e-4)
        supporting = set()
        for i, c in enumerate(d.contact):
            names = {m.geom(c.geom1).name, m.geom(c.geom2).name}
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if "ground" in names and force[0] > 0:
                supporting |= names - {"ground"}
        self.assertEqual(supporting, {f"{w}_link_contact" for w in WHEELS})
        METRICS["zero_torque_10_s"] = {
            "steps": round(10 / m.opt.timestep),
            "physics_warnings": int(sum(w.number for w in d.warning)),
            "max_ground_penetration_m": float(max_ground_penetration),
            "max_contact_penetration_m": float(max_penetration),
            "max_gripper_symmetry_error_m": float(max_gripper_error),
            "final_base_height_m": float(d.qpos[2]),
            "supporting_wheels": sorted(supporting),
            "actuator_torque_Nm": 0.0,
        }
        print(
            f"\n  10 s passive run: all four wheels support the robot; "
            f"max ground penetration={max_ground_penetration * 1000:.3f} mm; "
            f"max penetration overall={max_penetration * 1000:.3f} mm; "
            f"max finger symmetry error={max_gripper_error * 1000:.4f} mm; no warnings."
        )


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
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    height, width, _ = pixels.shape
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    print(f"Rendered {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        type=Path,
        help="Also save a PNG; requires OpenGL (MUJOCO_GL=egl on headless Linux)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save machine-readable results and model fingerprints",
    )
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.render:
        render(args.render)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "scene_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (MODEL_DIR / "source_manifest.json").read_bytes()
            ).hexdigest(),
            "metrics": METRICS,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
