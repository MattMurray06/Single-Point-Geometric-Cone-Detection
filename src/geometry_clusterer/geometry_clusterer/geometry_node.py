import numpy as np
import rclpy
from geometry_msgs.msg import Point
from perception_msgs.msg import ConeArrayWithConfidence
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray

from .cpd import CPD
from .centreline_voting import CentreLineVoting, CentreLineVotingResult
from .cone_detector import Cluster, ConeDetector
from .gaussian_propagation import GaussianPropagation
from .edge_propagation import EdgePropagation
from .centreline_propagation import CentreLinePropagation


class GeometryNode(Node):

    def __init__(self):
        super().__init__('geometry_node')

        self.get_logger().info('geometry node started')

        self.declare_parameter('detection_algorithm', 'cpd')
        self.declare_parameter('debug', False)
        self.declare_parameter('propagation_distance', 3.0)
        self.declare_parameter('track_width', 4.0)
        self.declare_parameter('sigma_longitudinal', 2.0)
        self.declare_parameter('sigma_lateral', 0.3)
        self.declare_parameter('max_mahalanobis_distance', 2.0)
        self.declare_parameter('score_threshold', 6.0)
        self.declare_parameter('max_neighbour_radius', 6.0)
        self.declare_parameter('min_neighbour_radius', 1.0)
        self.declare_parameter('edge_max_neighbour_radius', 6.0)
        self.declare_parameter('edge_min_neighbour_radius', 1.0)
        self.declare_parameter('max_angle_change', 30.0)
        self.declare_parameter('wall_density_threshold', 1.8)
        self.declare_parameter('max_consecutive_angle_change', 20.0)
        self.declare_parameter('beam_search_angle_deg', 30.0)
        self.declare_parameter('plot_growth', False)
        self.declare_parameter('min_track_width', 2.0)
        self.declare_parameter('max_track_width', 5.0)

        self.algorithm = self.get_parameter('detection_algorithm').get_parameter_value().string_value.lower()
        self.debug = self.get_parameter('debug').value
        self.propagation_distance = self.get_parameter(
            'propagation_distance'
        ).value
        self.track_width = self.get_parameter('track_width').value
        self.sigma_longitudinal = self.get_parameter('sigma_longitudinal').value
        self.sigma_lateral = self.get_parameter('sigma_lateral').value
        self.max_mahalanobis_distance = self.get_parameter(
            'max_mahalanobis_distance'
        ).value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.max_neighbour_radius = self.get_parameter(
            'max_neighbour_radius'
        ).value
        self.min_neighbour_radius = self.get_parameter(
            'min_neighbour_radius'
        ).value
        self.edge_max_neighbour_radius = self.get_parameter(
            'edge_max_neighbour_radius'
        ).value
        self.edge_min_neighbour_radius = self.get_parameter(
            'edge_min_neighbour_radius'
        ).value
        self.max_angle_change = self.get_parameter(
            'max_angle_change'
        ).value
        self.wall_density_threshold = self.get_parameter(
            'wall_density_threshold'
        ).value
        self.max_consecutive_angle_change = self.get_parameter(
            'max_consecutive_angle_change'
        ).value
        self.beam_search_angle_deg = self.get_parameter(
            'beam_search_angle_deg'
        ).value
        self.plot_growth = self.get_parameter('plot_growth').value
        self.min_track_width = self.get_parameter('min_track_width').value
        self.max_track_width = self.get_parameter('max_track_width').value

        self.detection_alg = self.create_detection_algorithm(
            self.algorithm,
            self.propagation_distance,
            self.track_width,
            self.sigma_longitudinal,
            self.sigma_lateral,
            self.max_mahalanobis_distance,
            self.score_threshold,
            self.max_neighbour_radius,
            self.min_neighbour_radius,
            self.edge_max_neighbour_radius,
            self.edge_min_neighbour_radius,
            self.max_angle_change,
            self.wall_density_threshold,
            self.max_consecutive_angle_change,
            self.beam_search_angle_deg,
            self.plot_growth,
            self.min_track_width,
            self.max_track_width,
        )

        self.cluster_info_sub_ = self.create_subscription(
            ConeArrayWithConfidence,
            "/cones",
            self.cluster_callback,
            QoSProfile(depth = 1)
        )

        self.cone_pub_ = self.create_publisher(
            MarkerArray,
            "/lidar/cones",
            QoSProfile(depth = 1)
        )

        self.centreline_pub_ = None
        if self.debug: 
            self.centreline_pub_ = self.create_publisher(
                MarkerArray,
                "/lidar/debug/centrelines",
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                )
            )

    def create_detection_algorithm(
        self,
        algorithm: str,
        propagation_distance: float,
        track_width: float,
        sigma_longitudinal: float,
        sigma_lateral: float,
        max_mahalanobis_distance: float,
        score_threshold: float,
        max_neighbour_radius: float,
        min_neighbour_radius: float,
        edge_max_neighbour_radius: float,
        edge_min_neighbour_radius: float,
        max_angle_change: float,
        wall_density_threshold: float,
        max_consecutive_angle_change: float,
        beam_search_angle_deg: float,
        plot_growth: bool,
        min_track_width: float,
        max_track_width: float,
    ) -> ConeDetector:
        algorithms: dict[str, type[ConeDetector]] = {
            'cpd': CPD,
            'voting': CentreLineVoting,
            'centreline propagation': CentreLinePropagation,
            'gaussian propagation': GaussianPropagation,
            'edge propagation': EdgePropagation,
        }

        algorithm_class = algorithms.get(algorithm)
        if algorithm_class is None:
            valid_algorithms = ', '.join(sorted(algorithms))
            raise ValueError(
                f'Unsupported detection_algorithm "{algorithm}". '
                f'Expected one of: {valid_algorithms}'
            )

        self.get_logger().info(f'Using {algorithm} detection algorithm')
        if algorithm == 'gaussian propagation':
            return GaussianPropagation(
                max_neighbour_radius=max_neighbour_radius,
                min_neighbour_radius=min_neighbour_radius,
                beam_search_angle_deg=beam_search_angle_deg,
                plot_growth=plot_growth,
            )
        if algorithm == 'edge propagation':
            return EdgePropagation(
                max_neighbour_radius=edge_max_neighbour_radius,
                min_neighbour_radius=edge_min_neighbour_radius,
                max_angle_change=max_angle_change,
                wall_density_threshold=wall_density_threshold,
                max_consecutive_angle_change=max_consecutive_angle_change,
                plot_growth=plot_growth,
                max_track_width=max_track_width,
                min_track_width=min_track_width,
            )
        if algorithm == 'voting':
            return CentreLineVoting(track_width=track_width)
        if algorithm == 'centreline propagation':
            return CentreLinePropagation(
                propagation_distance=propagation_distance,
                track_width=track_width,
                sigma_longitudinal=sigma_longitudinal,
                sigma_lateral=sigma_lateral,
                max_mahalanobis_distance=max_mahalanobis_distance,
                score_threshold=score_threshold,
            )
        return algorithm_class()

    def cluster_callback(self, clusters_msg):
        cluster_centres = self.ros_msg_to_array(clusters_msg)

        if isinstance(self.detection_alg, CentreLineVoting):
            detection_result = self.detection_alg.detect_cones(cluster_centres)
            if self.centreline_pub_ is not None:
                centreline_msg = self.centrelines_to_ros_msg(
                    detection_result.centre_lines, clusters_msg.header
                )

                self.centreline_pub_.publish(centreline_msg)

        elif isinstance(self.detection_alg, CentreLinePropagation):
            detection_result = self.detection_alg.detect_cones(cluster_centres)

            hypotheses = (
                detection_result
                if isinstance(detection_result, list)
                else [detection_result]
            )

            centrelines = [
                centreline
                for hypothesis in hypotheses
                for centreline in hypothesis.centrelines
            ]

            if self.centreline_pub_ is not None:
                centreline_msg = self.centrelines_to_ros_msg(
                    centrelines, clusters_msg.header
                )
                self.centreline_pub_.publish(centreline_msg)
        elif isinstance(self.detection_alg, EdgePropagation):
            detection_result = self.detection_alg.detect_cones(cluster_centres)
            if self.cone_pub_ is not None:
                cone_msg = self.array_to_ros_msg(detection_result, clusters_msg.header)

                self.cone_pub_.publish(cone_msg)

        else:
            detection_result = self.detection_alg.detect_cones(cluster_centres)
            cones = detection_result

        # cone_msg = self.array_to_ros_msg(cones, clusters_msg.header)

        # self.cone_pub_.publish(cone_msg)

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