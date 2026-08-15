#include "lidar_perception/lidar_pipeline.hpp"

#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <pcl/impl/point_types.hpp>

namespace lidar_perception
{

PipelineResult LidarPipeline::run_pipeline(const PointCloud::Ptr scan)
{
    PipelineResult result;
    auto roi_filtered = roi_filter(scan);
    result.roi_filtered = roi_filtered;
    auto ground_filtered = ground_removal(roi_filtered);
    result.ground_filtered = ground_filtered;
    auto downsampled = downsample(ground_filtered);
    result.downsampled = downsampled;
    auto clustered = cluster(downsampled);
    auto cones = post_process(clustered);
    //return cones;
    return result;
}

PointCloud LidarPipeline::roi_filter(
    const PointCloud::Ptr scan)
{
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(scan);

    crop.setMin(Eigen::Vector4f(0.0f, -8.0f, -2.0f, 1.0f));
    crop.setMax(Eigen::Vector4f(30.0f, 8.0f, 2.0f, 1.0f));

    PointCloud filtered;
    crop.filter(filtered);

    return filtered;
}

PointCloud LidarPipeline::ground_removal(
    const PointCloud& scan)
{
    return scan;
}

PointCloud LidarPipeline::downsample(const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    return scan;
}

std::vector<Eigen::Vector2f> LidarPipeline::cluster(const PointCloud& scan) {
    return {};
}

std::vector<Eigen::Vector2f> LidarPipeline::post_process(
    const std::vector<Eigen::Vector2f> clusters)
{
    return {};
}
}  // namespace lidar_perception