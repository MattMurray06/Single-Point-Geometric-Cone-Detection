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
		super().__init__('geometry_node') #Create ROS Node: geometry_node

		self.get_logger().info('geometry node started') 

		#ROS Node init parameters and their defaults (for params.yaml)
		self.declare_parameter('edge_max_neighbour_radius', 6.0)
		self.declare_parameter('edge_min_neighbour_radius', 1.0)
		self.declare_parameter('max_angle_change', 30.0)
		self.declare_parameter('wall_density_threshold', 1.8)
		self.declare_parameter('max_consecutive_angle_change', 20.0)
		self.declare_parameter('plot_growth', False)
		self.declare_parameter('min_track_width', 2.0)
		self.declare_parameter('max_track_width', 5.0)
		self.declare_parameter('min_segment_clusters', 6)

		#Init the filtering algo class
		param = lambda p: self.get_parameter(p).get_parameter_value()
		
		self.detection_alg = EdgePropagation(
			max_neighbour_radius=param('edge_max_neighbour_radius').double_value,
			min_neighbour_radius=param('edge_min_neighbour_radius').double_value,
			max_angle_change=param('max_angle_change').double_value,
			wall_density_threshold=param('wall_density_threshold').double_value,
			max_consecutive_angle_change=param('max_consecutive_angle_change').double_value,
			plot_growth=param('plot_growth').bool_value,
			max_track_width=param('max_track_width').double_value,
			min_track_width=param('min_track_width').double_value,
			min_segment_clusters=param('min_segment_clusters').integer_value,
		)

		#Subscribe to /cones and call receive_cone when a msg is received
		self.cluster_info_sub_ = self.create_subscription(
			ConeArrayWithConfidence,
			"/cones",
			self.receive_cone,
			QoSProfile(depth=1)
		)

		#Publish MarkerArray visualizer data to /lidar/cones
		self.cone_pub_ = self.create_publisher(
			MarkerArray,
			"/lidar/cones",
			QoSProfile(depth=1)
		)

	def receive_cone(self, incoming_msg):
		cluster_centres = self.msg_to_centres(incoming_msg)
		segments = self.detection_alg.detect_cones(cluster_centres)
		cones = [
			cluster
			for segment in segments
			for edge in (segment.edge_a, segment.edge_b)
			for cluster in edge.clusters
		]
		self.cone_pub_.publish(self.centres_to_msg(cones, incoming_msg.header))

	def msg_to_centres(self, msg: ConeArrayWithConfidence) -> list[Cluster]:
		return [Cluster(i, c.point.x, c.point.y, c.confident) for i, c in enumerate(msg.cones)]

	def centres_to_msg(self, centres: list[Cluster], header) -> MarkerArray:

		msg = MarkerArray()
		clear_marker = Marker()
		clear_marker.action = Marker.DELETEALL
		markers = [clear_marker]

		for marker_id, centre in enumerate(centres):
			marker = Marker()
			marker.header = header
			marker.ns = "geometry_cones"
			marker.id = marker_id
			marker.type = Marker.CYLINDER
			marker.action = Marker.ADD
			marker.pose.position.x = float(centre.x)
			marker.pose.position.y = float(centre.y)
			marker.pose.position.z = 0.25
			marker.pose.orientation.w = 1.0
			marker.scale.x = 0.3
			marker.scale.y = 0.3
			marker.scale.z = 0.5
			# Orange = confident cluster (many points), grey = sparse cluster
			if centre.confident:
				marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
			else:
				marker.color.r, marker.color.g, marker.color.b = 0.6, 0.6, 0.6
			marker.color.a = 1.0
			markers.append(marker)

		msg.markers = markers
		return msg


def main(args=None):
	rclpy.init(args=args)

	node = GeometryNode()

	rclpy.spin(node)

	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
