"""Verify the DH/mesh mapping independently against the source URDF chain."""

import unittest

import numpy as np
import trimesh
import uaibot as ub

from scripts.piper_uaibot import (
    HOME_Q,
    MODEL_DIR,
    NOMINAL_TCP_OFFSET,
    _primitive_collision,
    _rz,
    create_piper,
)


class PiperUaibotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.piper = create_piper()

    def test_all_link_frames_match_urdf_in_random_configurations(self):
        piper = self.piper
        rng = np.random.default_rng(42)
        configs = [np.zeros(6), HOME_Q]
        limits = piper.kinematics.joint_limits
        configs.extend(rng.uniform(limits[:, 0], limits[:, 1], size=(20, 6)))
        base = np.eye(4)
        base[:3, :3] = _rz(0.7)[:3, :3]
        base[:3, 3] = [0.4, -0.2, 0.15]
        for q in configs:
            h = base.copy()
            frames = piper.robot.fkm(q=piper.to_uaibot_q(q), axis="dh", htm=base, mode="python")
            for i, origin in enumerate(piper.kinematics.joint_origins):
                h = h @ origin @ _rz(q[i])
                np.testing.assert_allclose(
                    np.asarray(frames[i]) @ piper.kinematics.dh_to_link[i], h, atol=1e-9,
                )
            np.testing.assert_allclose(
                piper.robot.fkm(q=piper.to_uaibot_q(q), htm=base, mode="python"),
                h @ np.asarray(ub.Utils.trn([0, 0, NOMINAL_TCP_OFFSET])), atol=1e-9,
            )

    def test_tcp_approach_is_flange_positive_z(self):
        piper = self.piper
        q = piper.to_uaibot_q(HOME_Q)
        flange = np.asarray(piper.robot.fkm(q=q, axis="dh")[-1]) @ piper.kinematics.dh_to_link[-1]
        tcp = np.asarray(piper.robot.fkm(q=q))
        np.testing.assert_allclose(tcp[:3, 2], flange[:3, 2], atol=1e-10)
        # The native backend returns single-precision FK translations.
        np.testing.assert_allclose(tcp[:3, 3] - flange[:3, 3], NOMINAL_TCP_OFFSET * tcp[:3, 2], atol=1e-8)
        piper.set_configuration(HOME_Q)
        np.testing.assert_allclose(piper.eef_frame.htm, tcp, atol=1e-10)

    def test_collision_objects_are_registered_and_follow_links(self):
        piper = self.piper
        self.assertEqual([len(link.col_objects) for link in piper.robot.links], [1, 1, 1, 1, 1, 5])
        piper.set_configuration(HOME_Q)
        for frame, link in zip(piper.robot.fkm(axis="dh"), piper.robot.links):
            for obj, relative in link.col_objects:
                np.testing.assert_allclose(obj.htm, np.asarray(frame) @ relative, atol=1e-10)
        np.testing.assert_allclose(piper.base_collision.htm, piper.robot.htm @ piper.base_collision_pose)
        self.assertEqual(len(piper.dh_frames), 7)
        # Nonzero theta would be applied again in the JS mesh hierarchy.
        self.assertTrue(all(link.theta == 0 for link in piper.robot.links))
        for obj, frame in zip(piper.dh_frames[1:], piper.robot.fkm(axis="dh")):
            np.testing.assert_allclose(obj.htm, frame, atol=1e-10)

    def test_distance_uses_registered_primitives(self):
        piper = self.piper
        piper.set_configuration(HOME_Q)
        h = np.eye(4)
        h[:3, 3] = [1.0, 0.3, 0.5]
        obstacle = ub.Ball(name="test_obstacle", htm=h, radius=0.05)
        distances = piper.robot.compute_dist(obstacle)
        self.assertEqual(distances.no_items, 10)
        self.assertTrue(np.all(np.isfinite(distances.dist_vect)))
        self.assertTrue(np.all(distances.dist_vect > 0))
        # Moving the query object one meter away along x must increase distance.
        h[0, 3] += 1
        obstacle.set_ani_frame(h)
        farther = piper.robot.compute_dist(obstacle)
        self.assertGreater(float(np.min(farther.dist_vect)), float(np.min(distances.dist_vect)))

    def test_collision_envelopes_contain_source_hulls(self):
        names = ["piper_base_link", *[f"link{i}" for i in range(1, 7)],
                 "flange_link", "gripper_base", "gripper_link1", "gripper_link2"]
        for name in names:
            obj, pose = _primitive_collision(name, "test_" + name, "red", 0.3)
            vertices = trimesh.load_mesh(MODEL_DIR / f"assets/collision/{name}_hull.stl").vertices
            local = (vertices - pose[:3, 3]) @ pose[:3, :3]
            if isinstance(obj, ub.Cylinder):
                self.assertLessEqual(float(np.linalg.norm(local[:, :2], axis=1).max()), obj.radius)
                self.assertLessEqual(float(np.abs(local[:, 2]).max()), obj.height / 2)
            else:
                self.assertIsInstance(obj, ub.Box)
                self.assertTrue(np.all(np.abs(local).max(axis=0) <= np.array([obj.width, obj.depth, obj.height]) / 2))


if __name__ == "__main__":
    unittest.main()
