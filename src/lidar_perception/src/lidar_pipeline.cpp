#include "lidar_perception/lidar_pipeline.hpp"

#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <pcl/impl/point_types.hpp>

namespace lidar_perception
{

std::vector<Eigen::Vector2f> LidarPipeline::run_pipeline(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr scan)
{
    auto roi_filtered = roi_filter(scan);
    auto filtered = ground_removal(roi_filtered);
    auto downsampled = downsample(filtered);
    auto clustered = cluster(downsampled);
    auto cones = post_process(clustered);
    return cones;
}

pcl::PointCloud<pcl::PointXYZ> LidarPipeline::roi_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr scan)
{
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(scan);

    crop.setMin(Eigen::Vector4f(0.0f, -8.0f, -2.0f, 1.0f));
    crop.setMax(Eigen::Vector4f(30.0f, 8.0f, 2.0f, 1.0f));

    pcl::PointCloud<pcl::PointXYZ> filtered;
    crop.filter(filtered);

    return filtered;
}

pcl::PointCloud<pcl::PointXYZ> LidarPipeline::ground_removal(
    const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    return scan;
}

pcl::PointCloud<pcl::PointXYZ> LidarPipeline::downsample(const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    return scan;
}

std::vector<Eigen::Vector2f> LidarPipeline::cluster(const pcl::PointCloud<pcl::PointXYZ>& scan) {
    return {};
}

std::vector<Eigen::Vector2f> LidarPipeline::post_process(
    const std::vector<Eigen::Vector2f> clusters)
{
    return {};
}
}  // namespace lidar_perception