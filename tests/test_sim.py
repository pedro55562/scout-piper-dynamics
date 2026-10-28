"""Focused API and physical behavior checks; no controllers or modified physics."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

# Direct file execution adds tests/, not the repository root, to sys.path.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scout_piper_sim import ARM_JOINTS, WHEEL_JOINTS, ScoutPiperSim


def _yaw(quaternion):
    w, x, y, z = quaternion
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _contact_lateral_slip(sim):
    m, d = sim._model, sim._data
    lateral = d.xmat[sim._base_body].reshape(3, 3)[:, 1]
    peak = 0.0
    for contact in d.contact:
        if m.geom(contact.geom1).name == "ground":
            geom = contact.geom2
        elif m.geom(contact.geom2).name == "ground":
            geom = contact.geom1
        else:
            continue
        if not m.geom(geom).name.endswith("_wheel_link_contact"):
            continue
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, geom, velocity, 0)
        point_velocity = velocity[3:] + np.cross(
            velocity[:3], contact.pos - d.geom_xpos[geom]
        )
        normal = contact.frame[:3]
        tangent = lateral - normal * (normal @ lateral)
        tangent /= np.linalg.norm(tangent)
        peak = max(peak, abs(float(tangent @ point_velocity)))
    return peak


class SimTests(unittest.TestCase):
    def setUp(self):
        self.sim = ScoutPiperSim()

    def test_custom_reset_and_snapshots(self):
        s = self.sim
        conditions = {
            "base_q": [1, -2, 1.1],
            "base_dq": [0.3, -0.2, 0.4],
            "arm_q": [0.2, 0.8, -1.2, 0.1, 0.6, -0.2],
            "arm_dq": np.arange(6) * 0.03,
            "wheel_q": [0.1, 0.2, 0.3, 0.4],
            "wheel_dq": [1, 2, 3, 4],
        }
        s.set_torques(arm=np.ones(6), wheels=np.ones(4))
        s.reset(**conditions)
        state = s.get_state()
        self.assertEqual(state.time, 0)
        for group, size, prefix, names in [
            ("base", 3, "base", None),
            ("arm", 6, "arm", ARM_JOINTS),
            ("wheels", 4, "wheel", WHEEL_JOINTS),
        ]:
            part = getattr(state, group)
            np.testing.assert_allclose(part.q, conditions[prefix + "_q"], atol=1e-12)
            np.testing.assert_allclose(part.dq, conditions[prefix + "_dq"], atol=1e-12)
            for field in ["q", "dq", "ddq"]:
                array = getattr(part, field)
                self.assertEqual(array.shape, (size,))
                self.assertTrue(np.isfinite(array).all())
            if names:
                expected = [s._data.qacc[s._model.joint(n).dofadr[0]] for n in names]
                np.testing.assert_array_equal(part.ddq, expected)
        np.testing.assert_array_equal(s.arm_torques, 0)
        np.testing.assert_array_equal(s.wheel_torques, 0)
        self.assertEqual((s._model.nq, s._model.nv), (19, 18))
        self.assertEqual(
            s._data.qpos[s._base_qpos + 2], s._model.key("home").qpos[s._base_qpos + 2]
        )
        for part in [state.base, state.arm, state.wheels]:
            for field in ["q", "dq", "ddq"]:
                getattr(part, field)[:] = 999
        np.testing.assert_array_equal(s.get_state().arm.q, conditions["arm_q"])
        np.testing.assert_allclose(
            s.get_state().base.q, conditions["base_q"], atol=1e-12
        )
        s.reset()
        np.testing.assert_array_equal(s.get_state().arm.q, [0, 0.7, -1.4, 0, 0.7, 0])

    def test_commands_limits_and_validation(self):
        s = self.sim
        arm = [99, -99, 1, 99, -99, -1]
        wheels = [99, -99, 2, -2]
        s.set_torques(arm=arm, wheels=wheels)
        np.testing.assert_array_equal(
            s.arm_torques, np.clip(arm, -s.arm_torque_limits, s.arm_torque_limits)
        )
        np.testing.assert_array_equal(s.wheel_torques, [5, -5, 2, -2])
        s.set_arm_torques(np.zeros(6))
        np.testing.assert_array_equal(s.wheel_torques, [5, -5, 2, -2])
        s.set_wheel_torques(np.ones(4))
        applied = s.wheel_torques
        applied[:] = 100
        np.testing.assert_array_equal(s.wheel_torques, 1)
        with self.assertRaises(ValueError):
            s.set_torques(arm=np.ones(6), wheels=np.ones(3))
        np.testing.assert_array_equal(s.arm_torques, 0)  # Atomic validation.
        for bad in [np.ones((6, 1)), [np.nan] * 6, [np.inf] * 6]:
            with self.assertRaises(ValueError):
                s.set_arm_torques(bad)
        for bad in [-1, True, 1.5]:
            with self.assertRaises(ValueError):
                s.step(n=bad)
        before = s.get_state().base.q.copy()
        with self.assertRaises(ValueError):
            s.reset(base_q=[1, 2])
        np.testing.assert_array_equal(s.get_state().base.q, before)
        s.zero_torques()
        np.testing.assert_array_equal(s.arm_torques, 0)
        np.testing.assert_array_equal(s.wheel_torques, 0)
        custom = ScoutPiperSim(wheel_torque_limits=[1, 2, 3, 4])
        custom.set_wheel_torques([10, -10, 10, -10])
        np.testing.assert_array_equal(custom.wheel_torques, [1, -2, 3, -4])
        np.testing.assert_array_equal(custom.arm_torque_limits, s.arm_torque_limits)

    def test_tilted_heading_derivatives(self):
        s = self.sim
        for yaw in [-0.7, np.pi - 1e-5]:
            s.reset()
            m, d = s._model, s._data
            qx = np.array([np.cos(0.2), np.sin(0.2), 0, 0])
            qy = np.array([np.cos(0.25), 0, np.sin(0.25), 0])
            qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            temp = np.zeros(4)
            quaternion = np.zeros(4)
            mujoco.mju_mulQuat(temp, qz, qy)
            mujoco.mju_mulQuat(quaternion, temp, qx)
            p, v = s._base_qpos, s._base_dof
            d.qpos[p + 2] = 2
            d.qpos[p + 3 : p + 7] = quaternion
            d.qvel[v : v + 6] = [0.3, -0.2, 0.1, 0.7, -0.4, 1.2]
            d.qvel[s._arm_dof] = [0.1, 0.2, 0.3, -0.2, 0.1, 0.3]
            s._refresh()
            state = s.get_state().base
            h = 1e-4
            plus, minus = d.qpos.copy(), d.qpos.copy()
            mujoco.mj_integratePos(m, plus, d.qvel + 0.5 * h * d.qacc, h)
            mujoco.mj_integratePos(m, minus, d.qvel - 0.5 * h * d.qacc, -h)
            dp = np.angle(np.exp(1j * (_yaw(plus[p + 3 : p + 7]) - state.q[2])))
            dn = np.angle(np.exp(1j * (_yaw(minus[p + 3 : p + 7]) - state.q[2])))
            np.testing.assert_allclose(state.dq[:2], [0.3, -0.2], atol=1e-12)
            np.testing.assert_allclose(state.ddq[:2], d.qacc[v : v + 2], atol=1e-12)
            self.assertAlmostEqual(state.dq[2], (dp - dn) / (2 * h), delta=1e-7)
            self.assertAlmostEqual(state.ddq[2], (dp + dn) / h**2, delta=1e-5)
            world_omega = d.xmat[s._base_body].reshape(3, 3) @ d.qvel[v + 3 : v + 6]
            self.assertGreater(abs(state.dq[2] - world_omega[2]), 0.1)
        d.qpos[p + 3 : p + 7] = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
        s._refresh()
        with self.assertRaisesRegex(ValueError, "heading is undefined"):
            s.get_state()

    def test_determinism_batching_and_pure_reads(self):
        a, b = self.sim, ScoutPiperSim()
        for sim in [a, b]:
            sim.set_torques(arm=[0.1] * 6, wheels=[1] * 4)
        a.step(n=100)
        for _ in range(100):
            b.step()
            b.get_state()
            b.get_state()  # Reads must not influence physics.
        sa, sb = a.get_state(), b.get_state()
        self.assertEqual(sa.time, sb.time)
        for group in ["base", "arm", "wheels"]:
            for field in ["q", "dq", "ddq"]:
                np.testing.assert_array_equal(
                    getattr(getattr(sa, group), field),
                    getattr(getattr(sb, group), field),
                )
        a.step(n=0)
        self.assertEqual(a.get_state().time, sa.time)

    def test_all_ten_motors_produce_signed_motion(self):
        s = self.sim
        for group, n in [("arm", 6), ("wheels", 4)]:
            for i in range(n):
                for sign in [-1, 1]:
                    s.reset()
                    initial = getattr(s.get_state(), group).q[i]
                    torque = np.zeros(n)
                    torque[i] = sign
                    s.set_torques(**{group: torque})
                    s.step(n=5)
                    actual = getattr(s.get_state(), group)
                    self.assertGreater(
                        sign * (actual.q[i] - initial), 1e-7, (group, i, sign)
                    )
                    self.assertGreater(sign * actual.dq[i], 1e-6, (group, i, sign))

    def test_drive_turn_and_lateral_slip(self):
        s = self.sim
        for name, command, sign in [
            ("forward", [1] * 4, 1),
            ("backward", [-1] * 4, -1),
            ("left", [-5, 5, -5, 5], 1),
            ("right", [5, -5, 5, -5], -1),
        ]:
            s.reset()
            s.set_wheel_torques(command)
            headings = [s.get_state().base.q[2]]
            slip = 0
            for _ in range(700):
                s.step()
                headings.append(s.get_state().base.q[2])
                if name in ["left", "right"]:
                    slip = max(slip, _contact_lateral_slip(s))
            state = s.get_state().base
            heading = np.unwrap(headings)[-1]
            if name in ["forward", "backward"]:
                self.assertGreater(sign * state.q[0], 0.15)
                self.assertGreater(sign * state.dq[0], 0.2)
                self.assertLess(abs(heading), 0.05)
            else:
                self.assertGreater(sign * heading, 0.1)
                self.assertGreater(sign * state.dq[2], 0.1)
                self.assertGreater(slip, 0.02)
            print(
                f"\n  {name}: x={state.q[0]:.4f} m, yaw={heading:.4f} rad, lateral slip={slip:.4f} m/s"
            )

    def test_combined_motion_and_zero_torque_gravity(self):
        s = self.sim
        s.set_torques(arm=[0.2, 0.2, -0.1, 0.05, 0.03, 0.02], wheels=[1.5, 1, 1.5, 1])
        initial = s.get_state()
        s.step(n=1000)
        state = s.get_state()
        self.assertGreater(np.linalg.norm(state.base.q[:2] - initial.base.q[:2]), 0.1)
        self.assertGreater(np.linalg.norm(state.arm.q - initial.arm.q), 0.1)
        s.reset()
        s.zero_torques()
        initial = s.get_state()
        s.step(n=1000)
        self.assertGreater(np.max(abs(s.get_state().arm.q - initial.arm.q)), 0.1)
        np.testing.assert_array_equal(s._data.actuator_force, 0)
        np.testing.assert_array_equal(s._data.qfrc_actuator, 0)
        self.assertTrue(all(w.number == 0 for w in s._data.warning))


if __name__ == "__main__":
    unittest.main(verbosity=2)
