import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name="publisher_container",
            namespace="/",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="utils",
                    plugin="DatasetPublisher",
                    name="dataset_publisher",
                    parameters=["src/sara_utils/config/params.yaml"]
                )
            ]
        )
    ])