"""ROS 2 velocity controller and WebRTC keyboard teleoperation for Pegasus."""
from __future__ import annotations

import time

import carb
import numpy as np
import omni.appwindow
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

from flight_control_core import (
    VelocityCommand,
    quaternion_xyzw_to_matrix,
    velocity_control_wrench,
    yaw_from_matrix,
)
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend


CMD_VEL_TOPIC = "/drone0/cmd_vel"
ROTOR_TOPICS = tuple(f"/drone0/control/rotor{i}/ref" for i in range(4))


class FixedRotorROS2Backend(ROS2Backend):
    """Pegasus ROS backend with correctly captured rotor subscriber indices."""

    def __init__(self, *args, **kwargs):
        self.rotor_receive_counts = [0, 0, 0, 0]
        super().__init__(*args, **kwargs)

    def initialize_subscribers(self):
        if not self._sub_control:
            return
        self.rotor_subs = []
        for rotor_id in range(self._num_rotors):
            topic = f"{self._namespace}{self._id}/control/rotor{rotor_id}/ref"
            self.rotor_subs.append(
                self.node.create_subscription(
                    Float64,
                    topic,
                    lambda msg, index=rotor_id: self.rotor_callback(msg, index),
                    10,
                )
            )

    def rotor_callback(self, ros_msg: Float64, rotor_id):
        self.rotor_receive_counts[rotor_id] += 1
        super().rotor_callback(ros_msg, rotor_id)


class KeyboardRos2Teleop:
    """Translate WebRTC/Kit keyboard state into ROS 2 Twist messages."""

    def __init__(self, publisher, viewport, world_camera_path: str, drone_camera_path: str):
        self._publisher = publisher
        self._viewport = viewport
        self._world_camera_path = world_camera_path
        self._drone_camera_path = drone_camera_path
        self._using_drone_camera = False
        self._pressed: set[str] = set()
        self._movement_was_active = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    @property
    def using_drone_camera(self) -> bool:
        return self._using_drone_camera

    def _on_keyboard_event(self, event, *_) -> bool:
        key = event.input.name
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            if key == "C" and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self._using_drone_camera = not self._using_drone_camera
                self._viewport.camera_path = (
                    self._drone_camera_path
                    if self._using_drone_camera
                    else self._world_camera_path
                )
                carb.log_info(
                    "EUN camera view: "
                    + ("drone" if self._using_drone_camera else "world")
                )
            else:
                self._pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
        return True

    def publish(self) -> VelocityCommand | None:
        command = VelocityCommand(
            forward=float("W" in self._pressed) - float("S" in self._pressed),
            left=float("A" in self._pressed) - float("D" in self._pressed),
            up=float("R" in self._pressed) - float("F" in self._pressed),
            yaw_rate=float("Q" in self._pressed) - float("E" in self._pressed),
        ).clipped(2.0, 1.0, 1.0)
        movement_is_active = any(
            key in self._pressed for key in ("W", "S", "A", "D", "R", "F", "Q", "E")
        )
        if not movement_is_active and not self._movement_was_active:
            return None
        msg = Twist()
        msg.linear.x = command.forward
        msg.linear.y = command.left
        msg.linear.z = command.up
        msg.angular.z = command.yaw_rate
        self._publisher.publish(msg)
        self._movement_was_active = movement_is_active
        return command

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._subscription
            )
            self._subscription = None


class Ros2VelocityController(Backend):
    """Convert ROS 2 body velocity commands into ROS 2 rotor references."""

    def __init__(self, command_timeout: float = 0.6):
        super().__init__(None)
        self.node = None
        self._cmd_sub = None
        self._rotor_publishers = []
        self._keyboard_publisher = None
        self._keyboard = None
        self._command = VelocityCommand()
        self._command_timeout = command_timeout
        self._last_command_time = time.monotonic()
        self._state = None
        self._desired_yaw = 0.0
        self._yaw_initialized = False
        self._publish_accumulator = 0.0
        self._rotor_publish_accumulator = 0.0
        self._last_rotor_reference = np.zeros(4)
        self.cmd_vel_receive_count = 0

    def attach_ros_node(self, node) -> None:
        """Attach control endpoints to the Pegasus-owned ROS 2 node."""
        self.node = node
        self._cmd_sub = node.create_subscription(
            Twist, CMD_VEL_TOPIC, self._on_cmd_vel, 10
        )
        self._rotor_publishers = [
            node.create_publisher(Float64, topic, 10) for topic in ROTOR_TOPICS
        ]
        self._keyboard_publisher = node.create_publisher(Twist, CMD_VEL_TOPIC, 10)

    def attach_keyboard(self, viewport, world_camera_path: str, drone_camera_path: str):
        if self._keyboard_publisher is None:
            raise RuntimeError("attach_ros_node must be called before attach_keyboard")
        self._keyboard = KeyboardRos2Teleop(
            self._keyboard_publisher,
            viewport,
            world_camera_path,
            drone_camera_path,
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.cmd_vel_receive_count += 1
        self._command = VelocityCommand(
            forward=msg.linear.x,
            left=msg.linear.y,
            up=msg.linear.z,
            yaw_rate=msg.angular.z,
        ).clipped(2.0, 1.0, 1.0)
        self._last_command_time = time.monotonic()

    def publish_command(self, command: VelocityCommand) -> None:
        msg = Twist()
        msg.linear.x = command.forward
        msg.linear.y = command.left
        msg.linear.z = command.up
        msg.angular.z = command.yaw_rate
        self._keyboard_publisher.publish(msg)

    @property
    def last_rotor_reference(self) -> list[float]:
        return [float(value) for value in self._last_rotor_reference]

    def update_state(self, state) -> None:
        self._state = state
        if not self._yaw_initialized:
            rotation = quaternion_xyzw_to_matrix(state.attitude)
            self._desired_yaw = yaw_from_matrix(rotation)
            self._yaw_initialized = True

    def update(self, dt: float) -> None:
        self._publish_accumulator += dt
        self._rotor_publish_accumulator += dt
        if self._keyboard is not None and self._publish_accumulator >= 0.05:
            self._keyboard.publish()
            self._publish_accumulator = 0.0
        if self._state is None or self.vehicle is None:
            return
        command = self._command
        if time.monotonic() - self._last_command_time > self._command_timeout:
            command = VelocityCommand()
        self._desired_yaw += command.yaw_rate * dt
        thrust, torque = velocity_control_wrench(
            command=command,
            attitude_xyzw=self._state.attitude,
            linear_velocity_world=self._state.linear_velocity,
            angular_velocity_body=self._state.angular_velocity,
            desired_yaw=self._desired_yaw,
        )
        rotor_reference = self.vehicle.force_and_torques_to_velocities(thrust, torque)
        self._last_rotor_reference = np.asarray(rotor_reference, dtype=float)
        if self._rotor_publish_accumulator < 0.02:
            return
        self._rotor_publish_accumulator = 0.0
        for publisher, value in zip(self._rotor_publishers, rotor_reference):
            msg = Float64()
            msg.data = float(value)
            publisher.publish(msg)

    def input_reference(self):
        return self._last_rotor_reference

    def update_sensor(self, sensor_type: str, data) -> None:
        pass

    def update_graphical_sensor(self, sensor_type: str, data) -> None:
        pass

    def start(self) -> None:
        self._last_command_time = time.monotonic()

    def stop(self) -> None:
        self._command = VelocityCommand()
        self._last_rotor_reference = np.zeros(4)

    def reset(self) -> None:
        self.stop()
        self._yaw_initialized = False

    def close(self) -> None:
        if self._keyboard is not None:
            self._keyboard.close()
