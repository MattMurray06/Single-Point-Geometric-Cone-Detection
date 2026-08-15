#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <memory>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "lidar_perception/lidar_pipeline.hpp"
#include "rclcpp/rclcpp.hpp"

namespace lidar_perception
{
class LidarNode : public rclcpp::Node
{
   public:
    LidarNode() : Node("lidar_node")
    {
        RCLCPP_INFO(this->get_logger(), "Lidar node started");
        this->declare_parameter("output_cone_topic", "cones/");
        this->declare_parameter("input_lidar_topic", "lidar_points/");
        output_cone_topic_ = this->get_parameter("output_cone_topic").as_string();
        input_lidar_topic_ = this->get_parameter("input_lidar_topic").as_string();

        lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_lidar_topic_, rclcpp::SensorDataQoS(),
            std::bind(&LidarNode::lidar_callback, this, std::placeholders::_1));

        cone_publisher_ = this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(
            output_cone_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }

   private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_publisher_;

    std::string input_lidar_topic_;
    std::string output_cone_topic_;

    LidarPipeline lidar_pipeline_;

    void lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    eufs_msgs::msg::ConeArrayWithCovariance to_cone_array_msg(
        const std::vector<Eigen::Vector2f>& cones, const std_msgs::msg::Header& header);
};

void LidarNode::lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::fromROSMsg(*msg, *cloud);

    auto cones = lidar_pipeline_.run_pipeline(cloud);

    auto cone_msg = to_cone_array_msg(cones, msg->header);

    cone_publisher_->publish(cone_msg);
}

eufs_msgs::msg::ConeArrayWithCovariance LidarNode::to_cone_array_msg(
    const std::vector<Eigen::Vector2f>& cones, const std_msgs::msg::Header& header)
{
    eufs_msgs::msg::ConeArrayWithCovariance msg;

    msg.header = header;

    for (const auto& cone : cones)
    {
        eufs_msgs::msg::ConeWithCovariance cone_msg;

        cone_msg.point.x = cone.x();
        cone_msg.point.y = cone.y();
        cone_msg.point.z = 0.0;

        cone_msg.covariance = {0.0, 0.0, 0.0, 0.0};

        msg.unknown_color_cones.push_back(cone_msg);
    }

    return msg;
}
}  // namespace lidar_perception

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<lidar_perception::LidarNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}