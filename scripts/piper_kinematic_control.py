"""Resolved-rate Piper control using uaibot's task_function and dp_inv."""

from dataclasses import dataclass

import numpy as np
import uaibot as ub

# A reachable demonstration target, specified independently of the initial q.
TARGET_Q = np.array([0.35, 1.15, -1.5, 0.25, 0.45, -0.25])
# Top of the large circle: nominal TCP at [0.30, 0, 0.62] m. The center is
# [0.30, 0, 0.42] m with radius 0.20 m, leaving margin at the wrist limits.
CIRCLE_Q0 = np.array([-3.124366436e-7, 1.264326338497334, -2.002579557907521,
                       6.780614683e-7, 0.738253165696615, -4.167252934e-7])


@dataclass
class CircularTrajectory:
    """Circle in world YZ, optionally following the arm's lateral heading."""

    start_pose: np.ndarray
    radius: float = 0.04
    period: float = 8.0
    start_phase: float = 0.0
    follow_heading: bool = False

    def __post_init__(self):
        self.start_pose = np.asarray(self.start_pose, dtype=float).copy()
        if self.start_pose.shape != (4, 4) or not np.all(np.isfinite(self.start_pose)):
            raise ValueError("start_pose must be a finite 4x4 pose")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive and finite")
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("period must be positive and finite")
        if not np.isfinite(self.start_phase):
            raise ValueError("start_phase must be finite")
        if self.follow_heading and abs(self.start_pose[0, 3]) < 1e-6:
            raise ValueError("heading tracking requires a nonzero world x coordinate")

    @property
    def center(self):
        return self.start_pose[:3, 3] - self.radius * np.array(
            [0, np.cos(self.start_phase), np.sin(self.start_phase)])

    def __call__(self, time):
        omega = 2 * np.pi / self.period
        phase = self.start_phase + omega * time
        pose = self.start_pose.copy()
        pose[:3, 3] = self.center + self.radius * np.array([0, np.cos(phase), np.sin(phase)])
        twist = np.array([0, -self.radius * omega * np.sin(phase),
                          self.radius * omega * np.cos(phase), 0, 0, 0])
        if self.follow_heading:
            x, y = pose[:2, 3]
            yaw = np.arctan2(y, x) - np.arctan2(self.start_pose[1, 3], self.start_pose[0, 3])
            c, s = np.cos(yaw), np.sin(yaw)
            pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.start_pose[:3, :3]
            twist[5] = (x * twist[1] - y * twist[0]) / (x * x + y * y)
        return pose, twist


def task_feedforward(robot, target, target_twist, *, q=None):
    """Return -partial(r)/partial(t) for a desired world [v_d, omega_d].

    Uaibot's axis errors are 1 - a_des.T a_actual, so their desired-motion
    feedforward is dot(omega_des cross a_des, a_actual), not omega_des itself.
    """
    twist = np.asarray(target_twist, dtype=float).reshape(-1)
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ValueError("target_twist must contain six finite world velocities")
    feedforward = np.zeros((6, 1))
    feedforward[:3, 0] = twist[:3]
    if np.any(twist[3:]):
        actual = np.asarray(robot.fkm(q=q))
        desired = np.asarray(target)
        for axis in range(3):
            feedforward[axis + 3, 0] = np.dot(
                np.cross(twist[3:], desired[:3, axis]), actual[:3, axis])
    return feedforward


def kinematic_velocity(robot, target, *, q=None, k=2.0, damping=1e-6, target_twist=None):
    """Return qdot, r and Jr, targeting total rdot = -k*r (k > 0).

    q and qdot use uaibot's full DH joint coordinates. The damping argument is
    exactly dp_inv's eps in (Jr.T Jr + eps I)^-1 Jr.T. Damping makes the task
    rate approximate near singular configurations.
    """
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive and finite")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    r, jac_r = robot.task_function(np.asmatrix(target), q=q)
    feedforward = (np.zeros((6, 1)) if target_twist is None
                   else task_feedforward(robot, target, target_twist, q=q))
    # rdot = Jr*qdot + partial_t(r). Cancel the target-motion contribution.
    task_rate = -k * r + feedforward
    qdot = ub.Utils.dp_inv(jac_r, eps=damping) @ task_rate
    return qdot, r, jac_r


@dataclass
class ControlResult:
    time: np.ndarray
    q_urdf: np.ndarray
    task: np.ndarray
    qdot: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    target_pose: np.ndarray
    eef_pose: np.ndarray
    target_twist: np.ndarray


def run_kinematic_control(piper, target, *, duration=10.0, dt=0.02, k=2.0,
                          damping=1e-6, max_speed=1.0):
    """Track a fixed pose or target(t)->(pose, world twist), with Euler steps.

    This is a kinematic demo, without contact dynamics or collision avoidance.
    A common velocity scaling limits all joints without changing direction;
    joint positions are clipped to the URDF limits after each integration step.
    """
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be positive and finite")
    if not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("max_speed must be positive and finite")
    steps = int(np.ceil(duration / dt))
    times = np.linspace(0, duration, steps + 1)
    q = np.asarray(piper.robot.q).reshape(6).copy()
    offset = piper.kinematics.dh[:, 0]
    piper.set_configuration(q - offset)
    states, tasks, velocities, position_errors, angle_errors = [], [], [], [], []
    target_history, actual_history, twist_history = [], [], []
    limits = np.asarray(piper.robot.joint_limit)
    for i, time in enumerate(times):
        desired, twist = target(float(time)) if callable(target) else (target, np.zeros(6))
        desired = np.asmatrix(desired)
        qdot, r, _ = kinematic_velocity(piper.robot, desired, q=q, k=k, damping=damping,
                                       target_twist=twist)
        velocity = np.asarray(qdot).reshape(6)
        velocity *= min(1.0, max_speed / max(float(np.max(np.abs(velocity))), 1e-12))
        actual = np.asarray(piper.robot.fkm(q=q))
        rotation_error = np.asarray(desired[:3, :3]).T @ actual[:3, :3]
        target_history.append(np.asarray(desired))
        actual_history.append(actual)
        twist_history.append(np.asarray(twist).reshape(6))
        states.append(q - offset)
        tasks.append(np.asarray(r).reshape(6))
        velocities.append(velocity.copy())
        position_errors.append(float(np.linalg.norm(r[:3])))
        angle_errors.append(float(np.arccos(np.clip((np.trace(rotation_error) - 1) / 2, -1, 1))))
        if i == steps:
            break
        next_q = np.clip(q + (times[i + 1] - time) * velocity, limits[:, 0], limits[:, 1])
        q = next_q
        piper.add_ani_frame(float(times[i + 1]), q - offset)
    return ControlResult(times, np.array(states), np.array(tasks), np.array(velocities),
                         np.array(position_errors), np.array(angle_errors),
                         np.array(target_history), np.array(actual_history), np.array(twist_history))
