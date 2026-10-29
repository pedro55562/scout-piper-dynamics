"""Validate the mobile assembly without invoking any controller."""

import unittest

import numpy as np
import uaibot as ub

from scripts.scout_piper_uaibot import create_scout_mini, create_scout_piper
from scripts.view_scout_piper_uaibot import create_scene


class ScoutPiperUaibotTests(unittest.TestCase):
    def test_straight_reverse_and_wheel_signs(self):
        scout = create_scout_mini(base_q=[0, 0, np.pi / 2])
        scout.set_velocity(0.2, 0)
        scout.step(2)
        np.testing.assert_allclose(scout.base_q, [0, 0.4, np.pi / 2], atol=1e-14)
        np.testing.assert_allclose(scout.wheel_q, np.full(4, 0.4 / 0.08))
        self.assertAlmostEqual(scout.track_width, 0.49, places=10)
        scout.set_velocity(-0.2, 0)
        scout.step(2)
        np.testing.assert_allclose(scout.base_q[:2], 0, atol=1e-14)
        np.testing.assert_allclose(scout.wheel_q, 0, atol=1e-14)

    def test_exact_arc_and_in_place_turn(self):
        scout = create_scout_mini()
        scout.set_velocity(0.3, 0.5)
        scout.step(2)
        np.testing.assert_allclose(scout.base_q, [0.6 * np.sin(1), 0.6 * (1 - np.cos(1)), 1])
        expected = np.array([0.3 - 0.5 * 0.49 / 2, 0.3 + 0.5 * 0.49 / 2] * 2) * 2 / 0.08
        np.testing.assert_allclose(scout.wheel_q, expected)
        scout.reset()
        scout.set_velocity(0, 0.5)
        scout.step(2)
        np.testing.assert_allclose(scout.base_q, [0, 0, 1])
        self.assertTrue(np.all(scout.wheel_q[[0, 2]] < 0))
        self.assertTrue(np.all(scout.wheel_q[[1, 3]] > 0))
        # Both sides use the proper URDF axis: rolling axes align with body +y.
        for origin, axis in zip(scout.wheel_origins, scout.wheel_axes):
            np.testing.assert_allclose(origin[:3, :3] @ axis, [0, 1, 0], atol=1e-10)

    def test_mount_tcp_and_all_geometry_follow_mobile_base(self):
        model = create_scout_piper(base_q=[0.2, -0.3, 0.4])
        q_arm = model.arm_q
        relative_tcp = np.linalg.inv(model.scout.htm) @ model.piper.robot.fkm()
        np.testing.assert_allclose(model.mount_htm[:3, 3], [0, 0, 0.05401])
        self.assertAlmostEqual(model.piper.robot.htm[2, 3], 0.236008)
        model.set_velocity(0.15, -0.3)
        for _ in range(20):
            model.step(0.05)
        np.testing.assert_allclose(model.arm_q, q_arm, atol=1e-12)
        np.testing.assert_allclose(model.piper.robot.htm, model.scout.htm @ model.mount_htm)
        np.testing.assert_allclose(model.piper.robot.fkm(), model.scout.htm @ relative_tcp, atol=1e-7)
        np.testing.assert_allclose(model.scout.chassis.htm, model.scout.htm)
        np.testing.assert_allclose(model.piper.eef_frame.htm, model.piper.robot.fkm())
        for link, frame in zip(model.piper.robot.links, model.piper.robot.fkm(axis="dh")):
            for obj, local in link.col_objects:
                np.testing.assert_allclose(obj.htm, frame @ local)
        for wheel, col, local in zip(model.scout.wheels, model.scout.collision_objects[1:],
                                      model.scout.collision_poses[1:]):
            np.testing.assert_allclose(col.htm, wheel.htm @ local)
            self.assertAlmostEqual(col.htm[2, 3] - col.radius, 0.001, places=9)
        q_next = q_arm.copy()
        q_next[0] += 0.1
        model.step(0.05, arm_q=q_next)
        np.testing.assert_allclose(model.arm_q, q_next, atol=1e-12)

    def test_reset_invalid_commands_and_state_copies(self):
        model = create_scout_piper()
        model.set_velocity(0.1, 0.2)
        model.step(0.5)
        before = model.scout.base_q
        for dt in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                model.step(dt)
        with self.assertRaises(ValueError):
            model.set_velocity(np.nan, 0)
        with self.assertRaises(ValueError):
            model.step(0.1, arm_q=[0, 1])
        np.testing.assert_array_equal(model.scout.base_q, before)
        state = model.scout.base_q
        state[:] = 100
        np.testing.assert_array_equal(model.scout.base_q, before)
        model.reset(base_q=[0.4, 0.2, -0.5], wheel_q=[1, 2, 3, 4])
        self.assertEqual(model.time, 0)
        np.testing.assert_array_equal(model.scout.command, [0, 0])
        np.testing.assert_array_equal(model.scout.wheel_q, [1, 2, 3, 4])
        model.step(0.1)
        np.testing.assert_allclose(model.scout.base_q, [0.4, 0.2, -0.5])

    def test_scene_hides_collisions_and_demo_keeps_arm_fixed(self):
        sim, model = create_scene("http://127.0.0.1:8002/models/scout_piper/assets")
        names = [obj.name for obj in sim.list_of_objects]
        self.assertFalse(any("_col_" in name or "_dh_" in name for name in names))
        self.assertEqual(len(model.scout.wheels), 4)
        self.assertAlmostEqual(model.time, 18, places=10)
        arm_frames = np.array([frame[17] for frame in model.piper.robot._frames])
        np.testing.assert_allclose(arm_frames, np.broadcast_to(arm_frames[0], arm_frames.shape))
        for obj in [model.scout.chassis, *model.scout.wheels, model.piper.eef_frame]:
            self.assertAlmostEqual(obj._max_time, model.time)
        other = ub.Simulation(show_world_frame=False)
        model.add_to_simulation(other, show_collision=True)
        self.assertTrue(any("_col_" in obj.name for obj in other.list_of_objects))


if __name__ == "__main__":
    unittest.main()
