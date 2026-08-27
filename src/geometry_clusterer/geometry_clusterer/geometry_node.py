import numpy as np
import rclpy
from geometry_msgs.msg import Point
from eufs_msgs.msg import ConeArrayWithCovariance
from rclpy.node import Node
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from .cpd import CPD
from .centreline_voting import CentreLineVoting, CentreLineVotingResult


class GeometryNode(Node):

    def __init__(self):
        super().__init__('geometry_node')

        self.get_logger().info('geometry node started')

        self.declare_parameter('detection_algorithm', 'cpd')
        algorithm = self.get_parameter('detection_algorithm').get_parameter_value().string_value.lower()
        self.declare_parameter('debug_geometry', False)
        debug_geometry = self.get_parameter('debug_geometry').value
        algorithms = {
            'cpd': CPD,
            'geometry': CentreLineVoting,
        }
        if algorithm not in algorithms:
            valid_algorithms = ', '.join(sorted(algorithms))
            raise ValueError(
                f'Unsupported detection_algorithm "{algorithm}". '
                f'Expected one of: {valid_algorithms}'
            )
        self.detection_alg = algorithms[algorithm]()
        self.get_logger().info(f'Using {algorithm} detection algorithm')

        self.debug_geometry = debug_geometry and algorithm == 'geometry'

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

        self.centreline_pub_ = None
        if self.debug_geometry:
            self.centreline_pub_ = self.create_publisher(
                MarkerArray,
                "/lidar/debug/centrelines",
                QoSProfile(depth=1)
            )

    def cluster_callback(self, clusters_msg):
        cluster_centres = self.ros_msg_to_array(clusters_msg)

        detection_result = self.detection_alg.detect_cones(cluster_centres)
        if isinstance(detection_result, CentreLineVotingResult):
            cones = detection_result.cones
            if self.debug_geometry:
                centreline_msg = self.centrelines_to_ros_msg(
                    detection_result.centre_lines, clusters_msg.header
                )
                self.centreline_pub_.publish(centreline_msg)
        else:
            cones = detection_result

        # cone_msg = self.array_to_ros_msg(cones, clusters_msg.header)

        # self.cone_pub_.publish(cone_msg)

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

    def centrelines_to_ros_msg(self, centrelines, header) -> MarkerArray:
        msg = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        msg.markers.append(clear_marker)

        for marker_id, centreline in enumerate(centrelines):
            direction = np.array([
                np.cos(centreline.heading),
                np.sin(centreline.heading),
            ])
            start = centreline.position - direction * 1.0
            end = centreline.position + direction * 1.0

            marker = Marker()
            marker.header = header
            marker.ns = "geometry_centrelines"
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            for point in (start, end):
                point_msg = Point()
                point_msg.x = float(point[0])
                point_msg.y = float(point[1])
                point_msg.z = 0.05
                marker.points.append(point_msg)

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