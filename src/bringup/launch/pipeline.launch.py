from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
import os

def generate_launch_description():
    pkg_utils_share = get_package_share_directory("sara_utils")
    config = os.path.join(pkg_utils_share, "config", "params.yaml")

    foxglove_launch = IncludeLaunchDescription(
    AnyLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('foxglove_bridge'),
            'launch',
            'foxglove_bridge_launch.xml'
        )
    ),
    launch_arguments={'port': '8765'}.items())

    lidar_launch = IncludeLaunchDescription(AnyLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('lidar_perception'),
            'launch',
            'lidar.launch.py'
        )
    ))

    geometry_launch = IncludeLaunchDescription(AnyLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('geometry_clusterer'),
            'launch',
            'geometry_clusterer.launch.py'
        )
    ), launch_arguments={"log_level": "info"}.items())


    return LaunchDescription([
        Node(
            package='sara_utils',
            executable='dataset_publisher',
            parameters=[config],
        ),
        lidar_launch,
        foxglove_launch, 
        geometry_launch
    ])
