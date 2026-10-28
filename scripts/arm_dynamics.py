"""Shared MJCF configuration for ideal, joint-side Piper torque inputs."""

import xml.etree.ElementTree as ET

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))


def configure_arm_dynamics(root, source_parameters):
    """Configure six torque ports without retaining uncalibrated arm friction.

    source_parameters maps each joint name to its original effort, damping and
    friction values. Returning that provenance keeps rebuilds and the delivered
    scene consistent without changing the geometry or inertial data.
    """
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.Element("actuator")
        keyframe = root.find("keyframe")
        root.insert(
            list(root).index(keyframe) if keyframe is not None else len(root), actuators
        )
    records = {}
    for name in ARM_JOINTS:
        joint = root.find(f"worldbody/.//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Missing Piper joint: {name}")
        # Ideal arm joints: no springs, viscous/Coulomb friction, or invented rotor inertia.
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
        joint.set("stiffness", "0")
        joint.set("armature", "0")
        source = source_parameters[name]
        effort = float(source.get("command_limit_Nm", source["effort_Nm"]))
        if effort <= 0:
            raise ValueError(f"Invalid source effort limit for {name}: {effort}")
        motor_name = f"{name}_motor"
        for old in list(actuators):
            if old.get("joint") == name or old.get("name") == motor_name:
                actuators.remove(old)
        ET.SubElement(
            actuators,
            "motor",
            name=motor_name,
            joint=name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"{-effort:.12g} {effort:.12g}",
        )
        records[name] = {
            "actuator": motor_name,
            "gear": 1.0,
            "command_unit": "N*m",
            "ctrlrange_Nm": [-effort, effort],
            "source": dict(source),
            "damping_Nm_s_per_rad": 0.0,
            "frictionloss_Nm": 0.0,
            "stiffness_Nm_per_rad": 0.0,
            "armature_kg_m2": 0.0,
        }
    keyframe = root.find("keyframe")
    if keyframe is not None:
        for key in keyframe.findall("key"):
            key.set("ctrl", "0 0 0 0 0 0")
    return {
        "description": "Ideal joint-side torque motors; standard Piper DEFAULT SDK command envelope; no passive joint losses.",
        "gravity_m_per_s2": [0.0, 0.0, -9.81],
        "gravity_compensation": False,
        "joints": records,
    }
