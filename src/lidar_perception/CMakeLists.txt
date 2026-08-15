cmake_minimum_required(VERSION 3.8)
project(lidar_perception)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eufs_msgs REQUIRED)

# LiDAR processing library
add_library(lidar_pipeline
src/lidar_pipeline.cpp
)

target_include_directories(lidar_pipeline
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)
  
target_link_libraries(lidar_pipeline
  PUBLIC
    Eigen3::Eigen
    ${PCL_LIBRARIES}
)

# ROS2 Plumbing - ROS2 Node Executable
add_executable(lidar_node
  src/lidar_main.cpp
)

ament_target_dependencies(lidar_node
  rclcpp
  sensor_msgs
  pcl_conversions
  eufs_msgs
)

target_link_libraries(lidar_node
  lidar_pipeline
)
  
install(TARGETS
  lidar_node lidar_pipeline
  DESTINATION
  lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
