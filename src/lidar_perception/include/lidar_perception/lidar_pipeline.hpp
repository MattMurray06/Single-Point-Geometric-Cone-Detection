#include <eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace lidar_perception {

class lidar_pipeline {
    public:
        eufs_msgs::msg::ConeArrayWithCovariance run_pipeline(
            const pcl::PointCloud<pcl::PointXYZ> &scan);
    private:

        pcl::PointCloud<pcl::PointXYZ> ground_removal(
            const pcl::PointCloud<pcl::PointXYZ> &scan);
        
        pcl::PointCloud<pcl::PointXYZ> downsample(
            const pcl::PointCloud<pcl::PointXYZ> &scan);
};
} // namespace lidar_perception