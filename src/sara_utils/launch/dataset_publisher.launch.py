from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory("sara_utils")
    config = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        Node(
            package="sara_utils",
            executable="dataset_publisher",
            name="dataset_publisher",
            output="screen",
            parameters=[config],
        )
    ])
