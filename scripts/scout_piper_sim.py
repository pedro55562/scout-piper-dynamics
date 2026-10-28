"""Minimal joint-torque interface to the Scout Mini + Piper MuJoCo model."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/scout_piper/scene.xml"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
WHEEL_JOINTS = tuple(
    f"{side}_wheel" for side in ("front_left", "front_right", "rear_left", "rear_right")
)


@dataclass(frozen=True)
class JointState:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


@dataclass(frozen=True)
class SimState:
    base: JointState
    arm: JointState
    wheels: JointState
    time: float


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape ({size},) and contain finite numbers")
    return value.copy()


class ScoutPiperSim:
    """Perfect-state simulation, without controllers or estimation.

    base.q is world [x, y, heading], with heading wrapped to [-pi, pi]. Its
    derivatives describe that world configuration, including when tilted.
    The heading is undefined if the body's forward axis becomes vertical.
    All returned arrays are independent snapshots, not MuJoCo memory views.
    """

    def __init__(self, model_path=MODEL_PATH, *, wheel_torque_limits=None):
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._base_body = self._model.body("base_link").id
        base = self._model.joint("floating_base")
        self._base_qpos = int(base.qposadr[0])
        self._base_dof = int(base.dofadr[0])
        self._arm_qpos, self._arm_dof, self._arm_act = self._addresses(ARM_JOINTS)
        self._wheel_qpos, self._wheel_dof, self._wheel_act = self._addresses(
            WHEEL_JOINTS
        )
        if self._model.nu != 10:
            raise ValueError("ScoutPiperSim requires exactly ten torque motors")
        if wheel_torque_limits is not None:
            values = np.asarray(wheel_torque_limits, dtype=float)
            if values.ndim == 0:
                values = np.full(4, values)
            values = _vector(values, 4, "wheel_torque_limits")
            if np.any(values <= 0):
                raise ValueError("Wheel torque limits must be positive")
            self._model.actuator_ctrlrange[self._wheel_act] = np.column_stack(
                (-values, values)
            )
        self.reset()

    def _addresses(self, names):
        return (
            np.array([self._model.joint(name).qposadr[0] for name in names]),
            np.array([self._model.joint(name).dofadr[0] for name in names]),
            np.array([self._model.actuator(f"{name}_motor").id for name in names]),
        )

    @property
    def timestep(self):
        return float(self._model.opt.timestep)

    @property
    def arm_torque_limits(self):
        """Positive symmetric joint-side limits, shape (6,), N·m."""
        return self._model.actuator_ctrlrange[self._arm_act, 1].copy()

    @property
    def wheel_torque_limits(self):
        """Positive symmetric joint-side limits, FL/FR/RL/RR, N·m."""
        return self._model.actuator_ctrlrange[self._wheel_act, 1].copy()

    @property
    def arm_torques(self):
        """Current saturated commands, shape (6,), N·m."""
        return self._commands(self._arm_act)

    @property
    def wheel_torques(self):
        """Current saturated commands, FL/FR/RL/RR, N·m."""
        return self._commands(self._wheel_act)

    def _commands(self, ids):
        limits = self._model.actuator_ctrlrange[ids]
        return np.clip(self._data.ctrl[ids], limits[:, 0], limits[:, 1])

    def reset(
        self,
        *,
        base_q=None,
        base_dq=None,
        arm_q=None,
        arm_dq=None,
        wheel_q=None,
        wheel_dq=None,
    ):
        """Reset to home, replacing supplied planar/joint conditions; torques become zero."""
        inputs = [
            (base_q, 3, "base_q"),
            (base_dq, 3, "base_dq"),
            (arm_q, 6, "arm_q"),
            (arm_dq, 6, "arm_dq"),
            (wheel_q, 4, "wheel_q"),
            (wheel_dq, 4, "wheel_dq"),
        ]
        values = {
            name: _vector(value, size, name)
            for value, size, name in inputs
            if value is not None
        }
        m, d = self._model, self._data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        p, v = self._base_qpos, self._base_dof
        if "base_q" in values:
            x, y, heading = values["base_q"]
            d.qpos[p : p + 2] = [x, y]
            # Keep home's nominal z; custom planar reset starts level.
            d.qpos[p + 3 : p + 7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
        if "base_dq" in values:
            dx, dy, dheading = values["base_dq"]
            # Free-joint translation is world-frame; angular velocity is body-frame.
            d.qvel[v : v + 6] = [dx, dy, 0, 0, 0, dheading]
        for name, address in [
            ("arm_q", self._arm_qpos),
            ("arm_dq", self._arm_dof),
            ("wheel_q", self._wheel_qpos),
            ("wheel_dq", self._wheel_dof),
        ]:
            if name in values:
                (d.qpos if name.endswith("_q") else d.qvel)[address] = values[name]
        d.ctrl[:] = 0
        self._refresh()

    def set_arm_torques(self, arm_tau):
        self.set_torques(arm=arm_tau)

    def set_wheel_torques(self, wheel_tau):
        self.set_torques(wheels=wheel_tau)

    def set_torques(self, *, arm=None, wheels=None):
        """Set constant joint-side torques, saturating per motor. Omitted groups stay unchanged."""
        commands = []
        for value, ids, size, name in [
            (arm, self._arm_act, 6, "arm"),
            (wheels, self._wheel_act, 4, "wheels"),
        ]:
            if value is not None:
                tau = _vector(value, size, name)
                limits = self._model.actuator_ctrlrange[ids]
                commands.append((ids, np.clip(tau, limits[:, 0], limits[:, 1])))
        for ids, tau in commands:
            self._data.ctrl[ids] = tau
        self._refresh()

    def zero_torques(self):
        self._data.ctrl[:] = 0
        self._refresh()

    def _refresh(self):
        # mj_step integrates qpos/qvel after evaluating forces. Refresh at the new
        # configuration so snapshots' q, dq, ddq all refer to the same instant.
        mujoco.mj_forward(self._model, self._data)

    def step(self, n=1):
        """Advance n fixed steps; commands persist and no state is teleported."""
        if (
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
        ):
            raise ValueError("n must be a nonnegative integer")
        for _ in range(n):
            mujoco.mj_step(self._model, self._data)
            self._refresh()
            if any(w.number for w in self._data.warning) or not all(
                np.isfinite(a).all()
                for a in (self._data.qpos, self._data.qvel, self._data.qacc)
            ):
                raise RuntimeError(
                    "MuJoCo physics warning or nonfinite simulation state"
                )

    def get_state(self):
        d = self._data
        p, v = self._base_qpos, self._base_dof
        R = d.xmat[self._base_body].reshape(3, 3)
        forward = R[:, 0]
        horizontal = float(forward[:2] @ forward[:2])
        if horizontal < 1e-12:
            raise ValueError(
                "Planar heading is undefined: the base forward axis is vertical"
            )
        omega = R @ d.qvel[v + 3 : v + 6]
        alpha = R @ d.qacc[v + 3 : v + 6]
        df = np.cross(omega, forward)
        ddf = np.cross(alpha, forward) + np.cross(omega, df)
        cross = forward[0] * df[1] - forward[1] * df[0]
        heading_dq = cross / horizontal
        heading_ddq = (
            forward[0] * ddf[1] - forward[1] * ddf[0]
        ) / horizontal - 2 * cross * (forward[:2] @ df[:2]) / horizontal**2
        base = JointState(
            np.array([d.qpos[p], d.qpos[p + 1], np.arctan2(forward[1], forward[0])]),
            np.array([d.qvel[v], d.qvel[v + 1], heading_dq]),
            np.array([d.qacc[v], d.qacc[v + 1], heading_ddq]),
        )

        def joints(qpos, dof):
            return JointState(
                d.qpos[qpos].copy(), d.qvel[dof].copy(), d.qacc[dof].copy()
            )

        return SimState(
            base,
            joints(self._arm_qpos, self._arm_dof),
            joints(self._wheel_qpos, self._wheel_dof),
            float(d.time),
        )

    def launch_viewer(self, **kwargs):
        """Return MuJoCo's passive viewer handle; call handle.sync() in the simulation loop."""
        from mujoco import viewer

        return viewer.launch_passive(self._model, self._data, **kwargs)
