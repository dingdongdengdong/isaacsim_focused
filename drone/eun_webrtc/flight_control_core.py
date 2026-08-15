"""Pure math helpers for the EUN ROS 2 multirotor velocity controller."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity and yaw-rate command."""

    forward: float = 0.0
    left: float = 0.0
    up: float = 0.0
    yaw_rate: float = 0.0

    def clipped(
        self,
        max_horizontal_speed: float,
        max_vertical_speed: float,
        max_yaw_rate: float,
    ) -> "VelocityCommand":
        horizontal = np.array([self.forward, self.left], dtype=float)
        norm = float(np.linalg.norm(horizontal))
        if norm > max_horizontal_speed:
            horizontal *= max_horizontal_speed / norm
        return VelocityCommand(
            forward=float(horizontal[0]),
            left=float(horizontal[1]),
            up=float(np.clip(self.up, -max_vertical_speed, max_vertical_speed)),
            yaw_rate=float(np.clip(self.yaw_rate, -max_yaw_rate, max_yaw_rate)),
        )


def quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    """Convert an Isaac/Pegasus XYZW quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1.0e-9:
        return np.eye(3)
    x, y, z, w = np.array([x, y, z, w], dtype=float) / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def yaw_from_matrix(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def vee(skew_symmetric: np.ndarray) -> np.ndarray:
    return np.array(
        [-skew_symmetric[1, 2], skew_symmetric[0, 2], -skew_symmetric[0, 1]],
        dtype=float,
    )


def velocity_control_wrench(
    *,
    command: VelocityCommand,
    attitude_xyzw,
    linear_velocity_world,
    angular_velocity_body,
    desired_yaw: float,
    mass: float = 1.5,
    gravity: float = 9.81,
    velocity_gains=(4.0, 4.0, 6.0),
    attitude_gains=(3.5, 3.5, 2.0),
    angular_rate_gains=(0.55, 0.55, 0.35),
) -> tuple[float, np.ndarray]:
    """Return total body thrust and body torque for a velocity command."""
    rotation = quaternion_xyzw_to_matrix(attitude_xyzw)
    body_velocity_reference = np.array(
        [command.forward, command.left, command.up], dtype=float
    )
    world_velocity_reference = rotation @ body_velocity_reference
    velocity_error = world_velocity_reference - np.asarray(linear_velocity_world, dtype=float)

    desired_force = mass * (
        np.diag(velocity_gains) @ velocity_error + np.array([0.0, 0.0, gravity])
    )
    force_norm = float(np.linalg.norm(desired_force))
    if force_norm < 1.0e-6:
        desired_force = np.array([0.0, 0.0, mass * gravity])
        force_norm = float(np.linalg.norm(desired_force))

    desired_body_z = desired_force / force_norm
    desired_heading = np.array([np.cos(desired_yaw), np.sin(desired_yaw), 0.0])
    desired_body_y = np.cross(desired_body_z, desired_heading)
    desired_body_y_norm = float(np.linalg.norm(desired_body_y))
    if desired_body_y_norm < 1.0e-6:
        desired_body_y = np.array([-np.sin(desired_yaw), np.cos(desired_yaw), 0.0])
    else:
        desired_body_y /= desired_body_y_norm
    desired_body_x = np.cross(desired_body_y, desired_body_z)
    desired_rotation = np.column_stack(
        (desired_body_x, desired_body_y, desired_body_z)
    )

    rotation_error = 0.5 * vee(
        desired_rotation.T @ rotation - rotation.T @ desired_rotation
    )
    desired_angular_velocity = np.array([0.0, 0.0, command.yaw_rate])
    angular_velocity_error = (
        np.asarray(angular_velocity_body, dtype=float) - desired_angular_velocity
    )
    torque = -(
        np.diag(attitude_gains) @ rotation_error
        + np.diag(angular_rate_gains) @ angular_velocity_error
    )
    thrust = max(0.0, float(desired_force @ rotation[:, 2]))
    return thrust, torque
