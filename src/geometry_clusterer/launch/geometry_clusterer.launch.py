import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("geometry_clusterer")
    params_file = os.path.join(package_share, "config", "params.yaml")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info"),
            Node(
                package="geometry_clusterer",
                executable="geometry_node",
                name="geometry_node",
                output="screen",
                arguments=["--ros-args", "--log-level", log_level],
                parameters=[params_file],
            ),
        ]
    )
