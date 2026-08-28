#pragma once
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <Eigen/Core>
#include <vector>

namespace lidar_perception
{

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

struct ClusterInfo
{
    std::size_t point_count;
    Eigen::Vector3f position;
    float dx;
    float dy;
    float dz;
    float max_z;
    bool high_confidence;
};

struct PipelineResult
{
    PointCloud roi_filtered;
    PointCloud ground_filtered;
    pcl::ModelCoefficients ground_plane;
    PointCloud downsampled;
    std::vector<pcl::PointIndices> clusters;
    std::vector<ClusterInfo> cluster_info;
};

struct PipelineConfig
{
    Eigen::Vector4f detection_roi_min{0.0f, -40.0f, -0.65f, 1.0f};
    Eigen::Vector4f detection_roi_max{40.0f, 40.0f, 2.0f, 1.0f};
    Eigen::Vector4f ground_model_roi_min{0.0f, -15.0f, -0.65f, 1.0f};
    Eigen::Vector4f ground_model_roi_max{15.0f, 15.0f, 2.0f, 1.0f};
    Eigen::Vector4f car_exclusion_min{-1.0f, -1.0f, -10.0f, 1.0f};
    Eigen::Vector4f car_exclusion_max{1.0f, 1.0f, 10.0f, 1.0f};
    double cluster_tolerance{0.3};
    int min_cluster_size{1};
    int max_cluster_size{300};
    int min_point_count_for_confident{4};
    float orange_cone_dx{0.228f};
    float orange_cone_dy{0.228f};
    float orange_cone_dz{0.325f};
    float ground_z{-0.62f};
    float gate_tolerance{0.1f};
    double ground_distance_threshold{0.1};
    double ground_removal_threshold{0.1};
    int ground_max_iterations{300};
    double ground_max_angle_radians{0.2617993877991494};
};

class LidarPipeline
{
   public:
    PipelineResult run_pipeline(const PointCloud::ConstPtr& scan);

    void configure(const PipelineConfig& config);

   private:
    PipelineConfig config_;

    PointCloud::Ptr roi_filter(const PointCloud::ConstPtr& scan, const Eigen::Vector4f& min_bounds,
                               const Eigen::Vector4f& max_bounds);

    PointCloud::Ptr ground_removal(const PointCloud::ConstPtr& scan,
                                   pcl::ModelCoefficients& coefficients);

    PointCloud::ConstPtr downsample(const PointCloud::ConstPtr& scan);

    std::vector<pcl::PointIndices> cluster(const PointCloud::ConstPtr& scan);

    std::vector<ClusterInfo> extract_cluster_info(
        const std::vector<pcl::PointIndices>& clusters, const PointCloud::ConstPtr& scan);

    std::vector<ClusterInfo> filter_clusters(
        const std::vector<ClusterInfo>& cluster_info);

    std::vector<Eigen::Vector2f> post_process(
        const std::vector<ClusterInfo>& cluster_info);

    bool outside_ground_roi(float x, float y);

    pcl::ModelCoefficients calculate_ground_plane(const PointCloud::ConstPtr& scan);

    PointCloud::Ptr remove_ground_plane(const PointCloud::ConstPtr scan,
                                        const pcl::ModelCoefficients& coefficients);
};
}  // namespace lidar_perception
