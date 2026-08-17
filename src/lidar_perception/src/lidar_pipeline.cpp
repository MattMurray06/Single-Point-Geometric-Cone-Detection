#include "lidar_perception/lidar_pipeline.hpp"

#include <pcl/PointIndices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Core>
#include <limits>
#include <memory>

namespace lidar_perception
{

PipelineResult LidarPipeline::run_pipeline(const PointCloud::Ptr scan)
{
    PipelineResult result;

    auto roi_filtered = roi_filter(scan);
    result.roi_filtered = *roi_filtered;
    auto ground_filtered = ground_removal(roi_filtered);
    result.ground_filtered = *ground_filtered;
    auto downsampled = downsample(ground_filtered);
    result.downsampled = *downsampled;
    auto clustered = cluster(downsampled);
    result.clusters = clustered;
    auto cluster_info = extract_cluster_info(clustered, downsampled);
    result.cluster_info = cluster_info;
    auto cones = post_process(cluster_info);
    return result;
}

PointCloud::Ptr LidarPipeline::roi_filter(const PointCloud::Ptr scan)
{
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(scan);

    crop.setMin(roi_min_);
    crop.setMax(roi_max_);

    PointCloud::Ptr filtered = std::make_shared<PointCloud>(); 
    crop.filter(*filtered);

    return filtered;
}

PointCloud::Ptr LidarPipeline::ground_removal(const PointCloud::Ptr scan)
{
    return scan;
}

PointCloud::Ptr LidarPipeline::downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr scan)
{
    return scan;
}

std::vector<pcl::PointIndices> LidarPipeline::cluster(const PointCloud::Ptr scan)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>
    );

    tree->setInputCloud(scan);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    ece.setInputCloud(scan);

    ece.setClusterTolerance(cluster_tolerance_);
    ece.setMinClusterSize(min_cluster_size_);
    ece.setMaxClusterSize(max_cluster_size_);
    ece.setSearchMethod(tree);

    ece.extract(cluster_indices);

    return cluster_indices;
}

std::vector<ClusterInfo> LidarPipeline::extract_cluster_info(
    const std::vector<pcl::PointIndices>& clusters, const PointCloud::Ptr scan)
{
    std::vector<ClusterInfo> cluster_info;
    cluster_info.reserve(clusters.size());

    for (const auto& cluster : clusters)
    {
        if (cluster.indices.empty())
        {
            continue;
        }

        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f min_point = Eigen::Vector3f::Constant(
            std::numeric_limits<float>::max());
        Eigen::Vector3f max_point = Eigen::Vector3f::Constant(
            std::numeric_limits<float>::lowest());

        for (const auto index : cluster.indices)
        {
            const auto& point = scan->points[index];
            const Eigen::Vector3f point_vector(point.x, point.y, point.z);
            position += point_vector;
            min_point = min_point.cwiseMin(point_vector);
            max_point = max_point.cwiseMax(point_vector);
        }

        position /= static_cast<float>(cluster.indices.size());
        const Eigen::Vector3f dimensions = max_point - min_point;

        cluster_info.push_back(ClusterInfo{cluster.indices.size(), position, dimensions.x(),
                                           dimensions.y(), dimensions.z()});
    }

    return cluster_info;
}

std::vector<Eigen::Vector2f> LidarPipeline::post_process(
    const std::vector<ClusterInfo>& cluster_info)
{
    
    return {};
}

void LidarPipeline::set_roi_bounds(float min_x, float min_y, float min_z, float max_x, float max_y,
                                   float max_z)
{
    roi_min_ = Eigen::Vector4f(min_x, min_y, min_z, 1.0f);
    roi_max_ = Eigen::Vector4f(max_x, max_y, max_z, 1.0f);
}

void LidarPipeline::set_cluster_parameters(double cluster_tolerance, int min_cluster_size,
                                           int max_cluster_size)
{
    cluster_tolerance_ = cluster_tolerance;
    min_cluster_size_ = min_cluster_size;
    max_cluster_size_ = max_cluster_size;
}

}  // namespace lidar_perception