import math
import unittest

import numpy as np

from flight_control_core import (
    VelocityCommand,
    quaternion_xyzw_to_matrix,
    velocity_control_wrench,
)


class FlightControlCoreTests(unittest.TestCase):
    def test_command_clips_horizontal_vector_and_other_axes(self):
        command = VelocityCommand(3.0, 4.0, 2.0, -3.0).clipped(2.0, 1.0, 1.0)
        self.assertAlmostEqual(math.hypot(command.forward, command.left), 2.0)
        self.assertEqual(command.up, 1.0)
        self.assertEqual(command.yaw_rate, -1.0)

    def test_identity_quaternion_gives_identity_matrix(self):
        np.testing.assert_allclose(
            quaternion_xyzw_to_matrix([0.0, 0.0, 0.0, 1.0]), np.eye(3)
        )

    def test_hover_wrench_balances_gravity(self):
        thrust, torque = velocity_control_wrench(
            command=VelocityCommand(),
            attitude_xyzw=[0.0, 0.0, 0.0, 1.0],
            linear_velocity_world=[0.0, 0.0, 0.0],
            angular_velocity_body=[0.0, 0.0, 0.0],
            desired_yaw=0.0,
        )
        self.assertAlmostEqual(thrust, 1.5 * 9.81)
        np.testing.assert_allclose(torque, np.zeros(3), atol=1.0e-9)

    def test_forward_command_tilts_toward_forward_motion(self):
        _, torque = velocity_control_wrench(
            command=VelocityCommand(forward=1.0),
            attitude_xyzw=[0.0, 0.0, 0.0, 1.0],
            linear_velocity_world=[0.0, 0.0, 0.0],
            angular_velocity_body=[0.0, 0.0, 0.0],
            desired_yaw=0.0,
        )
        # Positive body-Y pitch tilts the body +Z thrust vector toward world +X.
        self.assertGreater(torque[1], 0.0)


if __name__ == "__main__":
    unittest.main()
