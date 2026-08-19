"""Project-owned Pegasus graph for publishing the drone RGB camera to ROS 2."""
from __future__ import annotations

import numpy as np
from isaacsim.core.utils import stage
from isaacsim.core.utils.prims import is_prim_path_valid
from omni.isaac.sensor import Camera
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.graphs import Graph
from pegasus.simulator.logic.vehicles import Vehicle


class EUNROS2RGBCameraGraph(Graph):
    """Attach one RGB render product to Isaac Sim's ROS 2 image writer."""

    def __init__(self, camera_prim_path: str, config: dict | None = None) -> None:
        super().__init__(graph_type="EUNROS2RGBCameraGraph")
        config = config or {}
        self._camera_prim_path = camera_prim_path
        self._resolution = tuple(config.get("resolution", (640, 480)))
        self._namespace = config.get("namespace", "")
        self._base_topic = config.get("topic", "")
        self._tf_frame_id = config.get("tf_frame_id", "")
        self._writer = None

    def initialize(self, vehicle: Vehicle) -> None:
        frame_id = self._camera_prim_path.rpartition("/")[-1]
        namespace = self._namespace or f"/{vehicle.vehicle_name}"
        base_topic = self._base_topic or f"/{frame_id}"
        tf_frame_id = self._tf_frame_id or frame_id
        if not self._camera_prim_path.startswith("/"):
            self._camera_prim_path = f"{vehicle.prim_path}/{self._camera_prim_path}"

        self.camera = Camera(
            prim_path=self._camera_prim_path,
            position=np.array([0.30, 0.0, 0.0]),
            frequency=30.0,
            resolution=self._resolution,
            orientation=Rotation.identity().as_quat(),
        )
        self.camera.initialize()
        if not is_prim_path_valid(self._camera_prim_path):
            raise RuntimeError(f"RGB camera prim is invalid: {self._camera_prim_path}")

        # Isaac's ROS image writer consumes the camera render product directly.
        # This publishes live RGB frames; it does not generate a Replicator dataset.
        import omni.replicator.core as rep

        self._writer = rep.writers.get("LdrColorSDROS2PublishImage")
        self._writer.initialize(
            nodeNamespace=namespace,
            topicName=f"{base_topic}/rgb",
            frameId=tf_frame_id,
            queueSize=1,
        )
        self._writer.attach([self.camera._render_product_path])
        graph_path = f"{self._camera_prim_path}_pub"
        stage.get_current_stage().DefinePrim(graph_path, "Xform")
        super().initialize(graph_path)
