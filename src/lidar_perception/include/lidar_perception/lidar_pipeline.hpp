#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <vector>

namespace lidar_perception
{

class LidarPipeline
{
   public:
    std::vector<Eigen::Vector2f> run_pipeline(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

   private:
    pcl::PointCloud<pcl::PointXYZ> roi_filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan);

    pcl::PointCloud<pcl::PointXYZ> ground_removal(const pcl::PointCloud<pcl::PointXYZ>& scan);

    pcl::PointCloud<pcl::PointXYZ> downsample(const pcl::PointCloud<pcl::PointXYZ>& scan);

    std::vector<Eigen::Vector2f> cluster(const pcl::PointCloud<pcl::PointXYZ>& scan);

    std::vector<Eigen::Vector2f> post_process(const std::vector<Eigen::Vector2f> clusters);
};
}  // namespace lidar_perception