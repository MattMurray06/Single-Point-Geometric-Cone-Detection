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
#include <pcl/impl/point_types.hpp>

namespace lidar_perception
{

PipelineResult LidarPipeline::run_pipeline(const PointCloud::ConstPtr& scan)
{
    PipelineResult result;

    auto roi_filtered = roi_filter(scan, config_.detection_roi_min, config_.detection_roi_max);
    result.roi_filtered = *roi_filtered;
    auto ground_filtered = ground_removal(roi_filtered, result.ground_plane);
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

PointCloud::Ptr LidarPipeline::roi_filter(const PointCloud::ConstPtr& scan,
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
    car_crop.setMin(config_.car_exclusion_min);
    car_crop.setMax(config_.car_exclusion_max);
    car_crop.setNegative(true);

    PointCloud::Ptr car_filtered = std::make_shared<PointCloud>();
    car_crop.filter(*car_filtered);

    return car_filtered;
}

PointCloud::Ptr LidarPipeline::ground_removal(const PointCloud::ConstPtr& scan,
                                              pcl::ModelCoefficients& coefficients)
{
    coefficients = calculate_ground_plane(scan);

    auto filtered = remove_ground_plane(scan, coefficients);

    return filtered;
}

PointCloud::ConstPtr LidarPipeline::downsample(const PointCloud::ConstPtr& scan)
{
    return scan;
}

std::vector<pcl::PointIndices> LidarPipeline::cluster(const PointCloud::ConstPtr& scan)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>
    );

    tree->setInputCloud(scan);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    ece.setInputCloud(scan);

    ece.setClusterTolerance(config_.cluster_tolerance);
    ece.setMinClusterSize(config_.min_cluster_size);
    ece.setMaxClusterSize(config_.max_cluster_size);
    ece.setSearchMethod(tree);

    ece.extract(cluster_indices);

    return cluster_indices;
}

std::vector<ClusterInfo> LidarPipeline::extract_cluster_info(
    const std::vector<pcl::PointIndices>& clusters, const PointCloud::ConstPtr& scan)
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
        bool too_big = cluster.dx > config_.orange_cone_dx + config_.gate_tolerance ||
            cluster.dy > config_.orange_cone_dy + config_.gate_tolerance ||
            cluster.dz > config_.orange_cone_dz + config_.gate_tolerance;
        bool too_high = cluster.max_z > config_.ground_z + config_.orange_cone_dz + config_.gate_tolerance;

        if (too_big || too_high)
        {
            continue;
        }

        // Mark high confidence cones
        auto candidate = cluster;
        bool enough_points = candidate.point_count > 3;
        bool off_ground = candidate.max_z > config_.ground_z + config_.gate_tolerance;
        bool sufficiently_close = !outside_ground_roi(cluster.position.x(), cluster.position.y()); 

        candidate.high_confidence = enough_points && off_ground && sufficiently_close; 
        filtered_clusters.push_back(candidate);
    }
    return filtered_clusters;
}

std::vector<Eigen::Vector2f> LidarPipeline::post_process(
    const std::vector<ClusterInfo>& cluster_info)
{

    return {};
}

bool LidarPipeline::outside_ground_roi(const float x, const float y)
{
    return x > config_.ground_model_roi_max.x() || y > config_.ground_model_roi_max.y() || x < config_.ground_model_roi_min.x() || y < config_.ground_model_roi_min.y();
}

pcl::ModelCoefficients LidarPipeline::calculate_ground_plane(const PointCloud::ConstPtr& scan)
{
    auto inliers = std::make_shared<pcl::PointIndices>();
    auto coefficients= std::make_shared<pcl::ModelCoefficients>();
    // While we want to look in a very large area for cones, ground points are most prevalent in the surrounding area
    // Further, large objects like stands, barriers etc make ground removal more complicated
    // Therefore, we fit the ground plane to a restricted area
    auto ground_model_cloud =
        roi_filter(scan, config_.ground_model_roi_min, config_.ground_model_roi_max);

    pcl::SACSegmentation<pcl::PointXYZ> segmentation;

    segmentation.setInputCloud(ground_model_cloud);
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(config_.ground_distance_threshold);
    segmentation.setMaxIterations(config_.ground_max_iterations);
    // The barrier is very visible in some frames, and accounts for a substantial proportion of points.
    // Setting an axis the model should remain roughly perpendicular to prevents fitting a ground plane to this
    segmentation.setAxis(Eigen::Vector3f{0.0f, 0.0f, 1.0f});
    segmentation.setEpsAngle(config_.ground_max_angle_radians);

    segmentation.segment(*inliers, *coefficients);
    return *coefficients;
}

PointCloud::Ptr LidarPipeline::remove_ground_plane(const PointCloud::ConstPtr scan, const pcl::ModelCoefficients& coefficients)
{
    float a = coefficients.values[0];
    float b = coefficients.values[1];
    float c = coefficients.values[2];
    float d = coefficients.values[3];

    // the normal may not point upwards
    if (c < 0.0f)
    {
        a = -a;
        b = -b;
        c = -c;
        d = -d;
    }

    const float normal_norm = std::sqrt(a * a + b * b + c * c);

    auto filtered =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    filtered->reserve(scan->size());

    for (const auto& point : scan->points)
    {
        const float height =
            (a * point.x + b * point.y + c * point.z + d)
            / normal_norm;

        bool outside_removal_area = outside_ground_roi(point.x, point.y);
        bool is_ground = height < config_.ground_removal_threshold;

        if (outside_removal_area || !is_ground)
        {
            filtered->push_back(point);
        }
    }

    return filtered;
}

void LidarPipeline::configure(const PipelineConfig& config)
{
    config_ = config;
}

}  // namespace lidar_perception
