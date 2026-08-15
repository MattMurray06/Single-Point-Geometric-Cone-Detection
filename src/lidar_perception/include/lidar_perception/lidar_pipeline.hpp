#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <pcl/impl/point_types.hpp>
#include <vector>

namespace lidar_perception
{

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

struct PipelineResult
{
    PointCloud roi_filtered;
    PointCloud ground_filtered;
    PointCloud downsampled;
};

class LidarPipeline
{
   public:
    PipelineResult run_pipeline(const PointCloud::Ptr scan);

   private:
    PointCloud roi_filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    PointCloud ground_removal(const pcl::PointCloud<pcl::PointXYZ>& scan);

    PointCloud downsample(const pcl::PointCloud<pcl::PointXYZ>& scan);

    std::vector<Eigen::Vector2f> cluster(const PointCloud& scan);

    std::vector<Eigen::Vector2f> post_process(const std::vector<Eigen::Vector2f> clusters);
};
}  // namespace lidar_perception