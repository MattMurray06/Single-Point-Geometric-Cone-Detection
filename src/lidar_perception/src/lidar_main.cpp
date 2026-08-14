#include "rclcpp/rclcpp.hpp"
#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace lidar_perception {
class LidarNode : public rclcpp::Node {
  public:
    LidarNode() : Node("lidar_node") {
        RCLCPP_INFO(this->get_logger(), "Lidar node started");
        this->declare_parameter("output_cone_topic", "cones/");
        this->declare_parameter("input_lidar_topic", "lidar_points/");
        output_cone_topic_ =
            this->get_parameter("output_cone_topic").as_string();
        input_lidar_topic_ =
            this->get_parameter("input_lidar_topic").as_string();

        lidar_subscription_ =
            this->create_subscription<sensor_msgs::msg::PointCloud2>(
                input_lidar_topic_, rclcpp::SensorDataQoS(),
                std::bind(&LidarNode::lidar_callback, this,
                          std::placeholders::_1));

        cone_publisher_ =
            this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(
                output_cone_topic_,
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        lidar_subscription_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr
        cone_publisher_;

    std::string input_lidar_topic_;
    std::string output_cone_topic_;

    void lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        // process lidar msg
        // publish cone message
    }
};
} // namespace lidar_perception

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<lidar_perception::LidarNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}