"""Check task descent, TCP convergence and synchronized animation."""

import unittest

import numpy as np
import uaibot as ub
from scipy.spatial.transform import Rotation

from scripts.piper_kinematic_control import (
    CIRCLE_Q0,
    TARGET_Q,
    CircularTrajectory,
    kinematic_velocity,
    run_kinematic_control,
    task_feedforward,
)
from scripts.piper_uaibot import create_piper
from scripts.view_piper_uaibot import create_scene


class PiperControlTests(unittest.TestCase):
    def test_large_circle_reference_ball_and_complete_pointcloud(self):
        sim, _, result = create_scene("http://127.0.0.1:8000/models/scout_piper/assets")
        objects = {obj.name: obj for obj in sim.list_of_objects}
        marker = objects["reference_point"]
        cloud = objects["circle_path"]
        self.assertIsInstance(marker, ub.Ball)
        self.assertEqual(marker.color, "cyan")
        self.assertIsInstance(cloud, ub.PointCloud)
        points = cloud.points
        center = result.target_pose[0, :3, 3] - np.array([0, 0, 0.20])
        np.testing.assert_allclose(np.linalg.norm(points.T - center, axis=1), 0.20, atol=1e-10)
        self.assertGreater(np.ptp(points[1]), 0.399)
        self.assertGreater(np.ptp(points[2]), 0.399)
        np.testing.assert_allclose(points[:, 0], points[:, -1], atol=1e-10)
        # The marker follows the desired pose, including all animation frames.
        frames = np.array(marker._frames)
        np.testing.assert_allclose(frames[:, 0], result.time)
        np.testing.assert_allclose(frames[:, [4, 8, 12]], result.target_pose[:, :3, 3], atol=5.1e-5)
        np.testing.assert_allclose(marker.htm, result.target_pose[-1], atol=1e-10)
        self.assertLess(result.position_error.max(), 0.001)
        self.assertLess(result.orientation_error.max(), np.radians(1))

    def test_heading_angular_feedforward_matches_trajectory_rotation(self):
        piper = create_piper(q0=CIRCLE_Q0, show_dh=False)
        trajectory = CircularTrajectory(piper.robot.fkm(), radius=0.20, period=20,
                                         start_phase=np.pi / 2, follow_heading=True)
        time, step = 3.7, 1e-4
        before, _ = trajectory(time - step)
        after, _ = trajectory(time + step)
        _, twist = trajectory(time)
        angular_velocity = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec() / (2 * step)
        np.testing.assert_allclose(angular_velocity, twist[3:], atol=1e-8)

    def test_control_scene_has_no_dh_or_collision_objects(self):
        sim, piper, _ = create_scene("http://127.0.0.1:8000/models/scout_piper/assets", duration=0.02)
        names = [obj.name for obj in sim.list_of_objects]
        self.assertNotIn(piper.base_collision.name, names)
        self.assertFalse(any("_col_" in name or "_dh_" in name for name in names))
        self.assertIn(piper.eef_frame.name, names)
        self.assertIn("target_tcp", names)
        self.assertIn("circle_path", names)
        self.assertTrue(sim.show_world_frame)

    def test_feedforward_cancels_explicit_target_task_derivative(self):
        piper = create_piper(show_dh=False)
        robot = piper.robot
        target = np.asarray(robot.fkm(q=piper.to_uaibot_q(TARGET_Q)))
        twist = np.array([0.03, -0.02, 0.01, 0.15, -0.1, 0.07])
        step = 1e-4
        tasks = []
        for dt in [-step, step]:
            shifted = target.copy()
            shifted[:3, 3] += dt * twist[:3]
            shifted[:3, :3] = Rotation.from_rotvec(dt * twist[3:]).as_matrix() @ target[:3, :3]
            tasks.append(robot.task_function(np.asmatrix(shifted), mode="python")[0])
        explicit_derivative = (tasks[1] - tasks[0]) / (2 * step)
        np.testing.assert_allclose(task_feedforward(robot, target, twist),
                                   -explicit_derivative, atol=1e-6)

    def test_circle_geometry_velocity_and_tracking(self):
        piper = create_piper(show_dh=False)
        trajectory = CircularTrajectory(piper.robot.fkm())
        pose0, _ = trajectory(0)
        pose1, _ = trajectory(trajectory.period)
        np.testing.assert_allclose(pose1, pose0, atol=1e-12)
        for time in [0, 1.3, 4.7]:
            pose, twist = trajectory(time)
            self.assertAlmostEqual(np.linalg.norm(pose[:3, 3] - trajectory.center), trajectory.radius)
            step = 1e-5
            difference = (trajectory(time + step)[0][:3, 3] - trajectory(time - step)[0][:3, 3]) / (2 * step)
            np.testing.assert_allclose(difference, twist[:3], atol=1e-9)
        result = run_kinematic_control(piper, trajectory, duration=16)
        rms = np.sqrt(np.mean(result.position_error**2))
        self.assertLess(rms, 0.001)
        self.assertLess(result.position_error.max(), 0.001)
        self.assertLess(result.orientation_error.max(), np.radians(2))
        other = create_piper(show_dh=False)
        without_ff = run_kinematic_control(other, lambda t: (trajectory(t)[0], np.zeros(6)), duration=16)
        self.assertLess(rms, np.sqrt(np.mean(without_ff.position_error**2)) / 10)
        self.assertLess(np.linalg.norm(result.eef_pose[-1, :3, 3] - result.eef_pose[0, :3, 3]), 0.001)

    def test_control_descends_task_and_converges_to_reachable_tcp_pose(self):
        piper = create_piper(show_dh=False)
        target = piper.robot.fkm(q=piper.to_uaibot_q(TARGET_Q))
        velocity, r, jac_r = kinematic_velocity(piper.robot, target)
        self.assertLess(float((r.T @ jac_r @ velocity).item()), 0)
        result = run_kinematic_control(piper, target)
        self.assertLess(result.position_error[-1], 0.001)
        self.assertLess(result.orientation_error[-1], np.radians(1))
        self.assertLess(np.linalg.norm(result.task[-1]), np.linalg.norm(result.task[0]) / 100)
        self.assertLessEqual(float(np.abs(result.qdot).max()), 1.0 + 1e-12)
        limits = piper.kinematics.joint_limits
        self.assertTrue(np.all(result.q_urdf >= limits[:, 0]))
        self.assertTrue(np.all(result.q_urdf <= limits[:, 1]))
        np.testing.assert_allclose(piper.eef_frame.htm, piper.robot.fkm(), atol=1e-10)
        self.assertEqual(piper.dh_frames, [])


if __name__ == "__main__":
    unittest.main()
