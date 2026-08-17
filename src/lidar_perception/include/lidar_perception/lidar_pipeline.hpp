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
    float length;
    float width;
    float height;
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

    void set_roi_bounds(float min_x, float min_y, float min_z, float max_x, float max_y,
                        float max_z);

    void set_cluster_parameters(double cluster_tolerance, int min_cluster_size,
                                int max_cluster_size);

   private:
    Eigen::Vector4f roi_min_{-30.0f, -30.0f, -30.0f, 1.0f};
    Eigen::Vector4f roi_max_{30.0f, 30.0f, 30.0f, 1.0f};
    double cluster_tolerance_{0.3};
    int min_cluster_size_{1};
    int max_cluster_size_{300};

    PointCloud::Ptr roi_filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    PointCloud::Ptr ground_removal(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    PointCloud::Ptr downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    std::vector<pcl::PointIndices> cluster(const PointCloud::Ptr scan);

    std::vector<ClusterInfo> extract_cluster_info(
        const std::vector<pcl::PointIndices>& clusters, const PointCloud::Ptr scan);

    std::vector<Eigen::Vector2f> post_process(
        const std::vector<ClusterInfo>& cluster_info);
};
}  // namespace lidar_perception