#include "lidar_perception/lidar_pipeline.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Core>
#include <limits>
#include <memory>

namespace lidar_perception
{

PipelineResult LidarPipeline::run_pipeline(const PointCloud::Ptr scan)
{
    PipelineResult result;

    auto roi_filtered = roi_filter(scan, detection_roi_min_, detection_roi_max_);
    result.roi_filtered = *roi_filtered;
    auto ground_filtered = ground_removal(roi_filtered);
    result.ground_filtered = *ground_filtered;
    auto downsampled = downsample(ground_filtered);
    result.downsampled = *downsampled;
    auto clustered = cluster(downsampled);
    result.clusters = clustered;
    auto cluster_info = extract_cluster_info(clustered, downsampled);
    auto filtered_cluster_info = filter_clusters(cluster_info);
    result.cluster_info = filtered_cluster_info;
    auto cones = post_process(filtered_cluster_info);
    return result;
}

PointCloud::Ptr LidarPipeline::roi_filter(const PointCloud::Ptr scan,
                                          const Eigen::Vector4f& min_bounds,
                                          const Eigen::Vector4f& max_bounds)
{
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(scan);

    crop.setMin(min_bounds);
    crop.setMax(max_bounds);

    PointCloud::Ptr filtered = std::make_shared<PointCloud>(); 
    crop.filter(*filtered);

    pcl::CropBox<pcl::PointXYZ> car_crop;
    car_crop.setInputCloud(filtered);
    car_crop.setMin(car_exclusion_min_);
    car_crop.setMax(car_exclusion_max_);
    car_crop.setNegative(true);

    PointCloud::Ptr car_filtered = std::make_shared<PointCloud>();
    car_crop.filter(*car_filtered);

    return car_filtered;
}

PointCloud::Ptr LidarPipeline::ground_removal(const PointCloud::Ptr scan)
{
    auto ground_model_cloud =
        roi_filter(scan, ground_model_roi_min_, ground_model_roi_max_);
    auto inliers = std::make_shared<pcl::PointIndices>();
    auto coefficients = std::make_shared<pcl::ModelCoefficients>();

    pcl::SACSegmentation<pcl::PointXYZ> segmentation;

    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(0.1);
    segmentation.setMaxIterations(300);
    segmentation.setInputCloud(ground_model_cloud);
    segmentation.segment(*inliers, *coefficients);

    // The barrier is very visible in some frames, and accounts for a substantial proportion of points. 
    // Setting an axis the model should remain roughly perpendicular to prevents fitting a ground plane to this
    segmentation.setAxis(Eigen::Vector3f{0.0f, 0.0f, 1.0f});
    segmentation.setEpsAngle(0.2617); // 15 degrees in radians
    
    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(ground_model_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*filtered);

    return filtered;
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
                                           dimensions.y(), dimensions.z(), max_point.z()});
    }

    return cluster_info;
}

std::vector<ClusterInfo> LidarPipeline::filter_clusters(
    const std::vector<ClusterInfo>& cluster_info)
{
    std::vector<ClusterInfo> filtered_clusters; 
    for (const auto& cluster : cluster_info)
    {
        // Checks if 
        // a) cluster is within the size an orange cone might be
        // b) the cluster's top point isn't above where a normal orange cone would be
        if (cluster.dx > orange_cone_dx_ + gate_tolerance_ || cluster.dy > orange_cone_dy_ + gate_tolerance_ || cluster.dz > orange_cone_dz_ + gate_tolerance_ || cluster.max_z > ground_z_ + orange_cone_dz_ + gate_tolerance_)
        {
            continue;
        }

        // Mark high confidence cones
        auto candidate = cluster;
        candidate.high_confidence = candidate.point_count > 3 && candidate.max_z > ground_z_ + gate_tolerance_;
        filtered_clusters.push_back(candidate);
    }
    return filtered_clusters;
}

std::vector<Eigen::Vector2f> LidarPipeline::post_process(
    const std::vector<ClusterInfo>& cluster_info)
{
    
    return {};
}

void LidarPipeline::set_detection_roi_bounds(float min_x, float min_y, float min_z, float max_x,
                                             float max_y, float max_z)
{
    detection_roi_min_ = Eigen::Vector4f(min_x, min_y, min_z, 1.0f);
    detection_roi_max_ = Eigen::Vector4f(max_x, max_y, max_z, 1.0f);
}

void LidarPipeline::set_ground_model_roi_bounds(float min_x, float min_y, float min_z,
                                                float max_x, float max_y, float max_z)
{
    ground_model_roi_min_ = Eigen::Vector4f(min_x, min_y, min_z, 1.0f);
    ground_model_roi_max_ = Eigen::Vector4f(max_x, max_y, max_z, 1.0f);
}

void LidarPipeline::set_car_exclusion_bounds(float min_x, float min_y, float min_z, float max_x,
                                             float max_y, float max_z)
{
    car_exclusion_min_ = Eigen::Vector4f(min_x, min_y, min_z, 1.0f);
    car_exclusion_max_ = Eigen::Vector4f(max_x, max_y, max_z, 1.0f);
}

void LidarPipeline::set_cluster_parameters(double cluster_tolerance, int min_cluster_size,
                                           int max_cluster_size)
{
    cluster_tolerance_ = cluster_tolerance;
    min_cluster_size_ = min_cluster_size;
    max_cluster_size_ = max_cluster_size;
}

void LidarPipeline::set_orange_cone_dimensions(float dx, float dy, float dz)
{
    orange_cone_dx_ = dx;
    orange_cone_dy_ = dy;
    orange_cone_dz_ = dz;
}

void LidarPipeline::set_filter_parameters(float ground_z, float gate_tolerance)
{
    ground_z_ = ground_z;
    gate_tolerance_ = gate_tolerance;
}

}  // namespace lidar_perception
