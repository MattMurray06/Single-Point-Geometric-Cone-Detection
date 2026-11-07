import numpy as np
import rclpy
from perception_msgs.msg import ConeArrayWithConfidence
from rclpy.node import Node
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from .cone_detector import Cluster
from .edge_propagation import EdgePropagation


class GeometryNode(Node):

    def __init__(self):
        super().__init__('geometry_node')

        self.get_logger().info('geometry node started')

        self.declare_parameter('edge_max_neighbour_radius', 6.0)
        self.declare_parameter('edge_min_neighbour_radius', 1.0)
        self.declare_parameter('max_angle_change', 30.0)
        self.declare_parameter('wall_density_threshold', 1.8)
        self.declare_parameter('max_consecutive_angle_change', 20.0)
        self.declare_parameter('plot_growth', False)
        self.declare_parameter('min_track_width', 2.0)
        self.declare_parameter('max_track_width', 5.0)

        self.detection_alg = EdgePropagation(
            max_neighbour_radius=self.get_parameter('edge_max_neighbour_radius').value,
            min_neighbour_radius=self.get_parameter('edge_min_neighbour_radius').value,
            max_angle_change=self.get_parameter('max_angle_change').value,
            wall_density_threshold=self.get_parameter('wall_density_threshold').value,
            max_consecutive_angle_change=self.get_parameter('max_consecutive_angle_change').value,
            plot_growth=self.get_parameter('plot_growth').value,
            max_track_width=self.get_parameter('max_track_width').value,
            min_track_width=self.get_parameter('min_track_width').value,
        )

        self.cluster_info_sub_ = self.create_subscription(
            ConeArrayWithConfidence,
            "/cones",
            self.cluster_callback,
            QoSProfile(depth=1)
        )

        self.cone_pub_ = self.create_publisher(
            MarkerArray,
            "/lidar/cones",
            QoSProfile(depth=1)
        )

    def cluster_callback(self, clusters_msg):
        cluster_centres = self.ros_msg_to_array(clusters_msg)
        cones = self.detection_alg.detect_cones(cluster_centres)
        self.cone_pub_.publish(self.array_to_ros_msg(cones, clusters_msg.header))

    def ros_msg_to_array(self, msg: ConeArrayWithConfidence) -> list[Cluster]:
        return [
            Cluster(
                position=np.array([cone.point.x, cone.point.y]),
                confident=cone.confident,
            )
            for cone in msg.cones
        ]

    def array_to_ros_msg(self, cones: np.ndarray, header) -> MarkerArray:
        cones = np.asarray(cones)

        msg = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        msg.markers.append(clear_marker)

        for marker_id, (x, y) in enumerate(cones):
            marker = Marker()
            marker.header = header
            marker.ns = "cpd_cones"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.5
            if marker_id % 2 == 0:
                marker.color.r = 0.0
                marker.color.g = 0.35
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.85
                marker.color.b = 0.0
            marker.color.a = 1.0
            msg.markers.append(marker)

        return msg


def main(args=None):
    rclpy.init(args=args)

    node = GeometryNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
