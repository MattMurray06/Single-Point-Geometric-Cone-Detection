from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cpd_test",
                executable="cpd_node",
                name="cpd_node",
                output="screen",
            )
        ]
    )
