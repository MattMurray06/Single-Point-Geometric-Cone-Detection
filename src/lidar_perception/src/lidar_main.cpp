#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cstdint>
#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <eufs_msgs/msg/cone_with_covariance.hpp>
#include <eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp>
#include <eufs_msgs/msg/detail/cone_with_covariance__struct.hpp>
#include <limits>
#include <memory>
#include <rclcpp/logging.hpp>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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
        this->declare_parameter("output_ground_plane_topic", "debug/lidar/ground_plane");
        this->declare_parameter("output_downsampled_topic", "debug/lidar/downsampled");
        this->declare_parameter("output_cluster_topic", "debug/lidar/clusters");
        this->declare_parameter("output_cluster_info_topic", "debug/lidar/cluster_info");
        this->declare_parameter("output_filtered_cluster_topic", "debug/lidar/filtered_clusters");
        this->declare_parameter("output_cone_topic", "cones");
        this->declare_parameter("detection_roi_min_x", 0.0);
        this->declare_parameter("detection_roi_min_y", -40.0);
        this->declare_parameter("detection_roi_min_z", -0.65);
        this->declare_parameter("detection_roi_max_x", 40.0);
        this->declare_parameter("detection_roi_max_y", 40.0);
        this->declare_parameter("detection_roi_max_z", 2.0);
        this->declare_parameter("ground_model_roi_min_x", 0.0);
        this->declare_parameter("ground_model_roi_min_y", -15.0);
        this->declare_parameter("ground_model_roi_min_z", -0.65);
        this->declare_parameter("ground_model_roi_max_x", 15.0);
        this->declare_parameter("ground_model_roi_max_y", 15.0);
        this->declare_parameter("ground_model_roi_max_z", 2.0);
        this->declare_parameter("car_exclusion_min_x", -1.0);
        this->declare_parameter("car_exclusion_min_y", -1.0);
        this->declare_parameter("car_exclusion_min_z", -10.0);
        this->declare_parameter("car_exclusion_max_x", 1.0);
        this->declare_parameter("car_exclusion_max_y", 1.0);
        this->declare_parameter("car_exclusion_max_z", 10.0);
        this->declare_parameter("cluster_tolerance", 0.3);
        this->declare_parameter("min_cluster_size", 1);
        this->declare_parameter("max_cluster_size", 300);
        this->declare_parameter("min_point_count_for_confident", 4);
        this->declare_parameter("orange_cone_dx", 0.228);
        this->declare_parameter("orange_cone_dy", 0.228);
        this->declare_parameter("orange_cone_dz", 0.325);
        this->declare_parameter("ground_z", -0.62);
        this->declare_parameter("gate_tolerance", 0.1);
        this->declare_parameter("ground_distance_threshold", 0.1);
        this->declare_parameter("ground_removal_threshold", 0.1);
        this->declare_parameter("ground_max_iterations", 300);
        this->declare_parameter("ground_max_angle_radians", 0.2617993877991494);

        input_pointcloud_topic_ = this->get_parameter("input_pointcloud_topic").as_string();
        debug_ = this->get_parameter("debug").as_bool();
        output_roi_topic_ = this->get_parameter("output_roi_topic").as_string();
        output_ground_topic_ = this->get_parameter("output_ground_topic").as_string();
        output_ground_plane_topic_ =
            this->get_parameter("output_ground_plane_topic").as_string();
        output_downsampled_topic_ = this->get_parameter("output_downsampled_topic").as_string();
        output_cluster_topic_ = this->get_parameter("output_cluster_topic").as_string();
        output_cluster_info_topic_ = this->get_parameter("output_cluster_info_topic").as_string();
        output_filtered_cluster_topic_ =
            this->get_parameter("output_filtered_cluster_topic").as_string();
        output_cone_topic_ = this->get_parameter("output_cone_topic").as_string();

        PipelineConfig config;
        config.detection_roi_min = Eigen::Vector4f(
            static_cast<float>(this->get_parameter("detection_roi_min_x").as_double()),
            static_cast<float>(this->get_parameter("detection_roi_min_y").as_double()),
            static_cast<float>(this->get_parameter("detection_roi_min_z").as_double()), 1.0f);
        config.detection_roi_max = Eigen::Vector4f(
            static_cast<float>(this->get_parameter("detection_roi_max_x").as_double()),
            static_cast<float>(this->get_parameter("detection_roi_max_y").as_double()),
            static_cast<float>(this->get_parameter("detection_roi_max_z").as_double()), 1.0f);
        config.ground_model_roi_min = Eigen::Vector4f(
            static_cast<float>(this->get_parameter("ground_model_roi_min_x").as_double()),
            static_cast<float>(this->get_parameter("ground_model_roi_min_y").as_double()),
            static_cast<float>(this->get_parameter("ground_model_roi_min_z").as_double()), 1.0f);
        config.ground_model_roi_max = Eigen::Vector4f(
            static_cast<float>(this->get_parameter("ground_model_roi_max_x").as_double()),
            static_cast<float>(this->get_parameter("ground_model_roi_max_y").as_double()),
            static_cast<float>(this->get_parameter("ground_model_roi_max_z").as_double()), 1.0f);
        config.car_exclusion_min = Eigen::Vector4f(
            static_cast<float>(this->get_parameter("car_exclusion_min_x").as_double()),
            static_cast<float>(this->get_parameter("car_exclusion_min_y").as_double()),
            static_cast<float>(this->get_parameter("car_exclusion_min_z").as_double()), 1.0f);
        config.car_exclusion_max = Eigen::Vector4f(
            static_cast<float>(this->get_parameter("car_exclusion_max_x").as_double()),
            static_cast<float>(this->get_parameter("car_exclusion_max_y").as_double()),
            static_cast<float>(this->get_parameter("car_exclusion_max_z").as_double()), 1.0f);
        config.cluster_tolerance = this->get_parameter("cluster_tolerance").as_double();
        config.min_cluster_size =
            static_cast<int>(this->get_parameter("min_cluster_size").as_int());
        config.max_cluster_size =
            static_cast<int>(this->get_parameter("max_cluster_size").as_int());
        config.min_point_count_for_confident = static_cast<int>(
            this->get_parameter("min_point_count_for_confident").as_int());
        config.orange_cone_dx =
            static_cast<float>(this->get_parameter("orange_cone_dx").as_double());
        config.orange_cone_dy =
            static_cast<float>(this->get_parameter("orange_cone_dy").as_double());
        config.orange_cone_dz =
            static_cast<float>(this->get_parameter("orange_cone_dz").as_double());
        config.ground_z = static_cast<float>(this->get_parameter("ground_z").as_double());
        config.gate_tolerance =
            static_cast<float>(this->get_parameter("gate_tolerance").as_double());
        config.ground_distance_threshold =
            this->get_parameter("ground_distance_threshold").as_double();
        config.ground_removal_threshold =
            this->get_parameter("ground_removal_threshold").as_double();
        config.ground_max_iterations =
            static_cast<int>(this->get_parameter("ground_max_iterations").as_int());
        config.ground_max_angle_radians =
            this->get_parameter("ground_max_angle_radians").as_double();

        lidar_pipeline_.configure(config);

        lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_pointcloud_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
            std::bind(&LidarNode::lidar_callback, this, std::placeholders::_1));

        if (debug_)
        {
            roi_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                output_roi_topic_, rclcpp::SensorDataQoS());
            ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                output_ground_topic_, rclcpp::SensorDataQoS());
            ground_plane_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
                output_ground_plane_topic_, rclcpp::QoS(1));
            downsampled_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                output_downsampled_topic_, rclcpp::SensorDataQoS());
            cluster_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                output_cluster_topic_, rclcpp::QoS(1));
            filtered_cluster_publisher_ =
                this->create_publisher<visualization_msgs::msg::MarkerArray>(
                    output_filtered_cluster_topic_, rclcpp::QoS(1));
            RCLCPP_INFO(this->get_logger(), "Debug publishers created");
        }
        cluster_info_publisher_ = this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(
            output_cluster_info_topic_, rclcpp::QoS(1));
        cone_publisher_ = this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>(
            output_cone_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }

   private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr roi_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ground_plane_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_publisher_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cluster_info_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filtered_cluster_publisher_;

    std::string input_pointcloud_topic_;
    std::string output_roi_topic_;
    std::string output_ground_topic_;
    std::string output_ground_plane_topic_;
    std::string output_downsampled_topic_;
    std::string output_cluster_topic_;
    std::string output_cluster_info_topic_;
    std::string output_filtered_cluster_topic_;
    std::string output_cone_topic_;
    bool debug_;

    LidarPipeline lidar_pipeline_;

    void lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    void remove_zero_points(const PointCloud::Ptr& cloud);

    eufs_msgs::msg::ConeArrayWithCovariance to_cone_array_msg(
        const std::vector<Eigen::Vector2f>& cones, const std_msgs::msg::Header& header);

    visualization_msgs::msg::MarkerArray to_cluster_markers(
        const std::vector<pcl::PointIndices>& clusters, const PointCloud& point_cloud,
        const std_msgs::msg::Header& header);

    visualization_msgs::msg::MarkerArray to_filtered_cluster_markers(
        const std::vector<ClusterInfo>& clusters, const std_msgs::msg::Header& header);

    visualization_msgs::msg::Marker to_ground_plane_marker(
        const pcl::ModelCoefficients& coefficients, const std_msgs::msg::Header& header);

    sensor_msgs::msg::PointCloud2 to_ros_msg(const PointCloud& point_cloud,
                                             const std_msgs::msg::Header& header);

    eufs_msgs::msg::ConeArrayWithCovariance cluster_info_to_ros_msg(
        const std::vector<ClusterInfo>& info, const std_msgs::msg::Header& header);
};

void LidarNode::lidar_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    RCLCPP_DEBUG(this->get_logger(), "Lidar Callback Started");
    auto cloud = std::make_shared<PointCloud>();
    pcl::fromROSMsg(*msg, *cloud);

    // Many points returned at (0, 0, 0), approximately 80% for the JT128
    remove_zero_points(cloud);

    auto result = lidar_pipeline_.run_pipeline(cloud);

    if (debug_)
    {
        roi_publisher_->publish(to_ros_msg(result.roi_filtered, msg->header));
        ground_publisher_->publish(to_ros_msg(result.ground_filtered, msg->header));
        ground_plane_publisher_->publish(
            to_ground_plane_marker(result.ground_plane, msg->header));
        downsampled_publisher_->publish(to_ros_msg(result.downsampled, msg->header));
        RCLCPP_DEBUG(this->get_logger(),
                     "Original PointCloud size: %zu; ROI Filtered PointCloud size: %zu",
                     cloud->size(), result.roi_filtered.size());

        cluster_publisher_->publish(
            to_cluster_markers(result.clusters, result.downsampled, msg->header));
        filtered_cluster_publisher_->publish(
            to_filtered_cluster_markers(result.cluster_info, msg->header));
    }
    cluster_info_publisher_->publish(cluster_info_to_ros_msg(result.cluster_info, msg->header));

    // auto cone_msg = to_cone_array_msg(cones, msg->header);

    // cone_publisher_->publish(cone_msg);
}

void LidarNode::remove_zero_points(const PointCloud::Ptr& cloud)
{
    const std::size_t original_size = cloud->size();
    cloud->points.erase(
        std::remove_if(
            cloud->points.begin(), cloud->points.end(),
            [](const pcl::PointXYZ& point)
            {
                return point.x == 0.0f && point.y == 0.0f && point.z == 0.0f;
            }),
        cloud->points.end());
    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;

    RCLCPP_DEBUG(this->get_logger(), "Removed %zu zero points", original_size - cloud->size());
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

visualization_msgs::msg::MarkerArray LidarNode::to_cluster_markers(
    const std::vector<pcl::PointIndices>& clusters, const PointCloud& point_cloud,
    const std_msgs::msg::Header& header)
{
    visualization_msgs::msg::MarkerArray cluster_markers;

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    cluster_markers.markers.push_back(clear_marker);

    int marker_id = 0;
    for (const auto& cluster : clusters)
    {
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        float max_z = std::numeric_limits<float>::lowest();

        for (const auto index : cluster.indices)
        {
            const auto& point = point_cloud.points[index];
            min_x = std::min(min_x, point.x);
            min_y = std::min(min_y, point.y);
            min_z = std::min(min_z, point.z);
            max_x = std::max(max_x, point.x);
            max_y = std::max(max_y, point.y);
            max_z = std::max(max_z, point.z);
        }

        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "clusters";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = (min_x + max_x) / 2.0;
        marker.pose.position.y = (min_y + max_y) / 2.0;
        marker.pose.position.z = (min_z + max_z) / 2.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = std::max(max_x - min_x, 0.05f);
        marker.scale.y = std::max(max_y - min_y, 0.05f);
        marker.scale.z = std::max(max_z - min_z, 0.05f);
        marker.color.r = 1.0f;
        marker.color.g = 0.5f;
        marker.color.a = 0.4f;
        cluster_markers.markers.push_back(marker);
    }

    return cluster_markers;
}

visualization_msgs::msg::MarkerArray LidarNode::to_filtered_cluster_markers(
    const std::vector<ClusterInfo>& clusters, const std_msgs::msg::Header& header)
{
    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    int marker_id = 0;
    for (const auto& cluster : clusters)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "filtered_clusters";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = cluster.position.x();
        marker.pose.position.y = cluster.position.y();
        marker.pose.position.z = cluster.position.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        if (cluster.high_confidence)
        {
            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
        } else{
            marker.color.r = 1.0f;
            marker.color.g = 0.0f;
        }
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;
        markers.markers.push_back(marker);
    }

    return markers;
}

visualization_msgs::msg::Marker LidarNode::to_ground_plane_marker(
    const pcl::ModelCoefficients& coefficients, const std_msgs::msg::Header& header)
{
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "ground_plane";
    marker.id = 0;

    if (coefficients.values.size() != 4 || std::abs(coefficients.values[2]) < 1e-6f)
    {
        marker.action = visualization_msgs::msg::Marker::DELETE;
        return marker;
    }

    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.r = 0.1f;
    marker.color.g = 0.5f;
    marker.color.b = 1.0f;
    marker.color.a = 0.35f;

    const double min_x = this->get_parameter("detection_roi_min_x").as_double();
    const double min_y = this->get_parameter("detection_roi_min_y").as_double();
    const double max_x = this->get_parameter("detection_roi_max_x").as_double();
    const double max_y = this->get_parameter("detection_roi_max_y").as_double();
    const float a = coefficients.values[0];
    const float b = coefficients.values[1];
    const float c = coefficients.values[2];
    const float d = coefficients.values[3];

    auto point_on_plane = [a, b, c, d](double x, double y)
    {
        geometry_msgs::msg::Point point;
        point.x = x;
        point.y = y;
        point.z = -(a * x + b * y + d) / c;
        return point;
    };

    const auto bottom_left = point_on_plane(min_x, min_y);
    const auto bottom_right = point_on_plane(max_x, min_y);
    const auto top_right = point_on_plane(max_x, max_y);
    const auto top_left = point_on_plane(min_x, max_y);
    marker.points = {bottom_left, bottom_right, top_right, bottom_left, top_right, top_left};
    return marker;
}

sensor_msgs::msg::PointCloud2 LidarNode::to_ros_msg(const PointCloud& point_cloud,
                                                    const std_msgs::msg::Header& header)
{
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(point_cloud, msg);
    msg.header = header;
    return msg;
}

eufs_msgs::msg::ConeArrayWithCovariance LidarNode::cluster_info_to_ros_msg(
    const std::vector<ClusterInfo>& cluster_info, const std_msgs::msg::Header& header)
{
    eufs_msgs::msg::ConeArrayWithCovariance msg{};
    msg.header = header;
    msg.unknown_color_cones.reserve(cluster_info.size());
    for (const auto& info : cluster_info)
    {
        eufs_msgs::msg::ConeWithCovariance p;
        p.point.x = info.position.x();
        p.point.y = info.position.y();
        p.point.z = info.position.z();
        msg.unknown_color_cones.push_back(p);
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
