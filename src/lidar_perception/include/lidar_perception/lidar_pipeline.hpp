#pragma once
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
    PointCloud downsampled;
    std::vector<pcl::PointIndices> clusters;
    std::vector<ClusterInfo> cluster_info;
};

class LidarPipeline
{
   public:
    PipelineResult run_pipeline(const PointCloud::Ptr scan);

    void set_detection_roi_bounds(float min_x, float min_y, float min_z, float max_x,
                                  float max_y, float max_z);

    void set_ground_model_roi_bounds(float min_x, float min_y, float min_z, float max_x,
                                     float max_y, float max_z);

    void set_car_exclusion_bounds(float min_x, float min_y, float min_z, float max_x,
                                  float max_y, float max_z);

    void set_cluster_parameters(double cluster_tolerance, int min_cluster_size,
                                int max_cluster_size);

    void set_orange_cone_dimensions(float dx, float dy, float dz);

    void set_filter_parameters(float ground_z, float gate_tolerance);

   private:
    Eigen::Vector4f detection_roi_min_{0.0f, -40.0f, -0.65f, 1.0f};
    Eigen::Vector4f detection_roi_max_{40.0f, 40.0f, 2.0f, 1.0f};
    Eigen::Vector4f ground_model_roi_min_{0.0f, -15.0f, -0.65f, 1.0f};
    Eigen::Vector4f ground_model_roi_max_{15.0f, 15.0f, 2.0f, 1.0f};
    Eigen::Vector4f car_exclusion_min_{-1.0f, -1.0f, -10.0f, 1.0f};
    Eigen::Vector4f car_exclusion_max_{1.0f, 1.0f, 10.0f, 1.0f};
    double cluster_tolerance_{0.3};
    int min_cluster_size_{1};
    int max_cluster_size_{300};
    float orange_cone_dx_{0.228f};
    float orange_cone_dy_{0.228f};
    float orange_cone_dz_{0.325f};
    float ground_z_{-0.62f};
    float gate_tolerance_{0.1f};

    PointCloud::Ptr roi_filter(const PointCloud::Ptr scan, const Eigen::Vector4f& min_bounds,
                               const Eigen::Vector4f& max_bounds);

    PointCloud::Ptr ground_removal(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    PointCloud::Ptr downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    std::vector<pcl::PointIndices> cluster(const PointCloud::Ptr scan);

    std::vector<ClusterInfo> extract_cluster_info(
        const std::vector<pcl::PointIndices>& clusters, const PointCloud::Ptr scan);

    std::vector<ClusterInfo> filter_clusters(
        const std::vector<ClusterInfo>& cluster_info);

    std::vector<Eigen::Vector2f> post_process(
        const std::vector<ClusterInfo>& cluster_info);
};
}  // namespace lidar_perception
