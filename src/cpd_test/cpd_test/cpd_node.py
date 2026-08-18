import numpy as np
import rclpy
from eufs_msgs.msg import ConeArrayWithCovariance
from rclpy.node import Node
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from .cpd import CPD


class CPDNode(Node):

    def __init__(self):
        super().__init__('cpd_node')

        self.get_logger().info('CPD node started')

        self.cluster_info_sub_ = self.create_subscription(
            ConeArrayWithCovariance,
            "/debug/lidar/cluster_info",
            self.cluster_callback,
            QoSProfile(depth = 1)
        )

        self.cone_pub_ = self.create_publisher(
            MarkerArray,
            "/lidar/cones",
            QoSProfile(depth = 1)
        )

        self.cpd = CPD()

    def cluster_callback(self, clusters_msg):
        cluster_centres = self.ros_msg_to_array(clusters_msg)

        cones = self.cpd.run_cpd(cluster_centres)

        cone_msg = self.array_to_ros_msg(cones, clusters_msg.header)

        self.cone_pub_.publish(cone_msg)

    def ros_msg_to_array(self, msg: ConeArrayWithCovariance) -> np.ndarray:
        clusters = np.empty((len(msg.unknown_color_cones), 2))
        for i in range(len(msg.unknown_color_cones)):
            clusters[i, 0] = msg.unknown_color_cones[i].point.x
            clusters[i, 1] = msg.unknown_color_cones[i].point.y
        return clusters

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
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 1.0
            msg.markers.append(marker)

        return msg

def main(args=None):
    rclpy.init(args=args)

    node = CPDNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()