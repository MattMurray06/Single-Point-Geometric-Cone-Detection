cmake_minimum_required(VERSION 3.8)
project(sara_utils)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)
include_directories(${PCL_INCLUDE_DIRS})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------- dataset_publisher_node --------
add_library(dataset_publisher_component SHARED src/dataset_publisher_node.cpp)
ament_target_dependencies(
  dataset_publisher_component
  tf2_ros
  tf2_geometry_msgs
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  builtin_interfaces
  pcl_conversions
  image_transport)
target_link_libraries(dataset_publisher_component ${PCL_LIBRARIES})

# -------- register component --------
rclcpp_components_register_node(
  dataset_publisher_component
  PLUGIN "DatasetPublisher"
  EXECUTABLE dataset_publisher
)

# -------- install --------
ament_export_targets(export_dataset_publisher_component)
install(TARGETS dataset_publisher_component
        EXPORT export_dataset_publisher_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib
)

install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}/)

# -------- testing --------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
