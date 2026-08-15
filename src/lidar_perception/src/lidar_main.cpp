#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <memory>
#include <rclcpp/logging.hpp>
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
        this->declare_parameter("input_pointcloud_topic", "lidar_points");
        this->declare_parameter("debug", false);
        this->declare_parameter("output_roi_topic", "debug/lidar/roi_filtered");
        this->declare_parameter("output_ground_topic", "debug/lidar/ground_filtered");
        this->declare_parameter("output_downsampled_topic", "debug/lidar/downsampled");
        this->declare_parameter("output_cone_topic", "cones");

        input_pointcloud_topic_ = this->get_parameter("input_pointcloud_topic").as_string();
        debug_ = this->get_parameter("debug").as_bool();
        output_roi_topic_ = this->get_parameter("output_roi_topic").as_string();
        output_ground_topic_ = this->get_parameter("output_ground_topic").as_string();
        output_downsampled_topic_ = this->get_parameter("output_downsampled_topic").as_string();
        output_cone_topic_ = this->get_parameter("output_cone_topic").as_string();

        lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_pointcloud_topic_, rclcpp::SensorDataQoS(),
            std::bind(&LidarNode::lidar_callback, this, std::placeholders::_1));
        if (debug_)
        {
            roi_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                output_roi_topic_, rclcpp::SensorDataQoS());
            ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                output_ground_topic_, rclcpp::SensorDataQoS());
            downsampled_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                output_downsampled_topic_, rclcpp::SensorDataQoS());
            RCLCPP_INFO(this->get_logger(), "Debug publishers created");
        }
        cone_publisher_ = this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(
            output_cone_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }

   private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr roi_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;

    std::string input_pointcloud_topic_;
    std::string output_roi_topic_;
    std::string output_ground_topic_;
    std::string output_downsampled_topic_;
    std::string output_cone_topic_;
    bool debug_;

    LidarPipeline lidar_pipeline_;

    void lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    eufs_msgs::msg::ConeArrayWithCovariance to_cone_array_msg(
        const std::vector<Eigen::Vector2f>& cones, const std_msgs::msg::Header& header);
};

void LidarNode::lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    RCLCPP_DEBUG(this->get_logger(), "Lidar Callback Started");
    auto cloud = std::make_shared<PointCloud>();
    pcl::fromROSMsg(*msg, *cloud);

    auto result = lidar_pipeline_.run_pipeline(cloud);

    if (debug_)
    {
        sensor_msgs::msg::PointCloud2 roi_msg;
        pcl::toROSMsg(result.roi_filtered, roi_msg);
        roi_msg.header = msg->header;
        roi_publisher_->publish(roi_msg);

        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(result.ground_filtered, ground_msg);
        ground_msg.header = msg->header;
        ground_publisher_->publish(ground_msg);

        sensor_msgs::msg::PointCloud2 downsampled_msg;
        pcl::toROSMsg(result.downsampled, downsampled_msg);
        downsampled_msg.header = msg->header;
        downsampled_publisher_->publish(downsampled_msg);
    }

    // auto cone_msg = to_cone_array_msg(cones, msg->header);

    // cone_publisher_->publish(cone_msg);
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