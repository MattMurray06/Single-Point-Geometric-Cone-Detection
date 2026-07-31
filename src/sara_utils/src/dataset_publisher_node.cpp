#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <thread>
#include <atomic>
#include <algorithm>
#include <functional>
#include <cctype>
#include <cstring>
#include <sstream>
#include <iostream>

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <png.h>

#include "rclcpp/rclcpp.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "image_transport/image_transport.hpp"

#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2/convert.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/static_transform_broadcaster.h"




using namespace std::chrono_literals;
namespace fs = std::filesystem;

class DatasetPublisher : public rclcpp::Node
{
public:
  DatasetPublisher(const rclcpp::NodeOptions& options) : Node("dataset_publisher", options)
  {
    // Core parameters
    this->declare_parameter("mode", "sequential");  // sequential or manual
    this->declare_parameter("loop", false);
    this->declare_parameter("replay_rate", 1.0);

    this->declare_parameter("load_pcd_dir", "");
    this->declare_parameter("load_image_dir", "");
    this->declare_parameter("load_odometry_file", "");

    this->declare_parameter("output_cloud_topic", "/hesai/pcl");
    this->declare_parameter("output_camera_topic", "/camera/image");
    this->declare_parameter("output_lidar_timestamp_topic", "lidar/timestamp");
    this->declare_parameter("output_twist_topic", "/twist");

    this->declare_parameter("lidar_frame_id", "hesai_lidar");
    this->declare_parameter("camera_frame_id", "camera");
    this->declare_parameter("camera_optical_frame_id", "camera_optical");

    this->declare_parameter("log", true);
    this->get_parameter("log", should_log);

    // Optional camera intrinsics
    this->declare_parameter("fx", 0.0);
    this->declare_parameter("fy", 0.0);
    this->declare_parameter("cx", 0.0);
    this->declare_parameter("cy", 0.0);

    this->get_parameter("mode", mode_);
    this->get_parameter("loop", loop_);
    this->get_parameter("replay_rate", replay_rate_);

    this->get_parameter("lidar_frame_id", lidar_frame_id_);
    this->get_parameter("camera_frame_id", camera_frame_id_);
    this->get_parameter("camera_optical_frame_id", camera_optical_frame_id);

    const std::string pcd_dir = this->get_parameter("load_pcd_dir").as_string();
    const std::string image_dir = this->get_parameter("load_image_dir").as_string();
    const std::string odometry_file = this->get_parameter("load_odometry_file").as_string();
    const std::string output_cloud_topic = this->get_parameter("output_cloud_topic").as_string();
    const std::string output_camera_topic = this->get_parameter("output_camera_topic").as_string();
    const std::string output_twist_topic = this->get_parameter("output_twist_topic").as_string();
    const std::string output_lidar_timestamp_topic =
      this->get_parameter("output_lidar_timestamp_topic").as_string();

    fx_ = this->get_parameter("fx").as_double();
    fy_ = this->get_parameter("fy").as_double();
    cx_ = this->get_parameter("cx").as_double();
    cy_ = this->get_parameter("cy").as_double();

    if (replay_rate_ <= 0.0)
    {
      RCLCPP_WARN(this->get_logger(), "replay_rate must be > 0. Falling back to 1.0");
      replay_rate_ = 1.0;
    }

    cloud_publisher_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(output_cloud_topic, 5);

    //image_publisher_ =
    //    this->create_publisher<sensor_msgs::msg::Image>(output_camera_topic, rclcpp::SensorDataQoS());
    image_pub_ = image_transport::create_publisher(this, output_camera_topic, rmw_qos_profile_sensor_data);

    camera_info_publisher_ =
        this->create_publisher<sensor_msgs::msg::CameraInfo>(output_camera_topic + "_camera_info", 5);

    lidar_timestamp_publisher_ =
        this->create_publisher<builtin_interfaces::msg::Time>(output_lidar_timestamp_topic, 5);

    twist_publisher_ = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(output_twist_topic, 5);

    buildDataset(pcd_dir, image_dir, odometry_file);

    if (events_.empty())
    {
      RCLCPP_WARN(this->get_logger(), "No dataset events found.");
      rclcpp::shutdown();
      return;
    }

    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publishCameraOpticalTransform();

    RCLCPP_INFO(this->get_logger(),
                "Loaded %zu lidar frames, %zu camera frames, %zu total events.",
                lidar_events_.size(), image_events_.size(), events_.size());

    if (mode_ == "sequential")
    {
      RCLCPP_INFO(this->get_logger(),
                  "Sequential async replay active. replay_rate=%.3f loop=%s",
                  replay_rate_, loop_ ? "true" : "false");

      replay_thread_ = std::thread(&DatasetPublisher::replayLoop, this);
    }
    else
    {
      RCLCPP_INFO(this->get_logger(),
                  "Manual mode active. Press l for next lidar, c for next camera, q to quit.");

      input_thread_ = std::thread(&DatasetPublisher::keyboardListener, this);
    }

  }

  ~DatasetPublisher()
  {
    if (input_thread_.joinable())
      input_thread_.join();

    if (replay_thread_.joinable())
      replay_thread_.join();
  }


  std::string getMode() const
  {
    return mode_;
  }

private:
  enum class EventType
  {
    IMAGE,
    CLOUD,
    ODOMETRY
  };

  struct TimedEvent
  {
    long long timestamp_ns;
    EventType type;
    std::string path;
  };

  struct OdometryMessage
  {
    long long time;
    double x;
    double y;
    double z;
    double qx;
    double qy;
    double qz;
    double qw;
  };

  std::vector<TimedEvent> events_;
  std::vector<TimedEvent> lidar_events_;
  std::vector<TimedEvent> image_events_;
  std::vector<TimedEvent> odometry_events_;

  size_t next_lidar_i_{0};
  size_t next_image_i_{0};

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  //rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  image_transport::Publisher image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;
  rclcpp::Publisher<builtin_interfaces::msg::Time>::SharedPtr lidar_timestamp_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_publisher_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::thread input_thread_;
  std::thread replay_thread_;

  std::string mode_;
  bool loop_{false};
  double replay_rate_{1.0};

  bool should_log{true};

  std::string lidar_frame_id_;
  std::string camera_frame_id_;
  std::string odometry_frame_id_;
  std::string camera_optical_frame_id;

  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};

  std::string last_odometry_line_;

  static long long timestampToLong(const std::string& s)
  {
    try
    {
      return std::stoll(s);
    }
    catch (...)
    {
      return 0;
    }
  }

  static bool hasExtensionCaseInsensitive(const fs::path& path, const std::string& ext_expected)
  {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string expected = ext_expected;
    std::transform(expected.begin(), expected.end(), expected.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return ext == expected;
  }

  static bool isNewerEvent(const TimedEvent& a, const TimedEvent& b)
  {
    if (a.timestamp_ns != b.timestamp_ns)
      return a.timestamp_ns < b.timestamp_ns;

    // Stable tie-break so ordering is deterministic if timestamps are equal
    return static_cast<int>(a.type) < static_cast<int>(b.type);
  }

  void doesNotExistWarn(const std::string& path, const std::string& name) {
    if (path.empty() && !fs::exists(path)) {
      RCLCPP_WARN(this->get_logger(), "%s does not exist: %s", name.c_str(), path.c_str());
    }
  }

  

  void buildDataset(const std::string& pcd_dir, const std::string& image_dir, const std::string& odometry_file)
  {
    
    if (pcd_dir.empty() && image_dir.empty() && odometry_file.empty())
    {
      RCLCPP_WARN(this->get_logger(), "At least one of load_pcd_dir, load_image_dir or load_odometry_file must be set.");
      return;
    }

    doesNotExistWarn(pcd_dir, "PCD directory");
    doesNotExistWarn(image_dir, "Image directory");
    doesNotExistWarn(odometry_file, "Odometry file");


    if (!pcd_dir.empty() && fs::exists(pcd_dir))
    {
      for (const auto& entry : fs::recursive_directory_iterator(pcd_dir))
      {
        if (!entry.is_regular_file() || !hasExtensionCaseInsensitive(entry.path(), ".pcd"))
          continue;

        const long long ts = timestampToLong(entry.path().stem().string());
        if (ts <= 0)
          continue;

        TimedEvent ev{ts, EventType::CLOUD, entry.path().string()};
        lidar_events_.push_back(ev);
        events_.push_back(ev);
      }
    }

    if (!image_dir.empty() && fs::exists(image_dir))
    {
      for (const auto& entry : fs::recursive_directory_iterator(image_dir))
      {
        if (!entry.is_regular_file() || !hasExtensionCaseInsensitive(entry.path(), ".png"))
          continue;

        const long long ts = timestampToLong(entry.path().stem().string());
        if (ts <= 0)
          continue;

        TimedEvent ev{ts, EventType::IMAGE, entry.path().string()};
        image_events_.push_back(ev);
        events_.push_back(ev);
      }
    }

    if (!odometry_file.empty() && fs::exists(odometry_file))
    {
      if(should_log) { RCLCPP_INFO(this->get_logger(), "Odometry file exists and is called %s.", odometry_file); }
      auto path = fs::path(odometry_file);

      if (hasExtensionCaseInsensitive(path, ".csv")) {
        if(should_log) { RCLCPP_INFO(this->get_logger(), "Odometry file has .csv extension and name %s.", path.string()); }
        std::ifstream csv_file(path);
        std::string line;

        //Skip header line
        std::getline(csv_file, line);
        if(should_log) { RCLCPP_INFO(this->get_logger(), "Header line is %s.", line); }

        //Read each row of csv
        while (std::getline(csv_file, line)) {
          const long long ts = timestampToLong(getOdometryTime(line));
          if (ts <= 0)
            continue;
          TimedEvent ev{ts, EventType::ODOMETRY, line};
          odometry_events_.push_back(ev);
          events_.push_back(ev);
        }
      }
    }

    std::sort(lidar_events_.begin(), lidar_events_.end(), isNewerEvent);
    std::sort(image_events_.begin(), image_events_.end(), isNewerEvent);
    std::sort(odometry_events_.begin(), odometry_events_.end(), isNewerEvent);
    std::sort(events_.begin(), events_.end(), isNewerEvent);
  }

  std::string getOdometryTime(const std::string& odometry_line) {
    std::istringstream iss(odometry_line); // Create a string stream from the line
    std::string word; // Temporary string to hold extracted data

    std::getline(iss, word, ',');

    return word;
  }

  //TODO add odometry functionality to this if needed
  void keyboardListener()
  {
    int input_fd = open("/dev/tty", O_RDONLY | O_NOCTTY);

    if (input_fd == -1)
    {
      if (isatty(STDIN_FILENO))
        input_fd = STDIN_FILENO;
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Manual mode requires a TTY. Could not open /dev/tty.");
        rclcpp::shutdown();
        return;
      }
    }

    struct termios oldt, newt;
    if (tcgetattr(input_fd, &oldt) != 0)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to read terminal attributes.");
      if (input_fd != STDIN_FILENO)
        close(input_fd);
      rclcpp::shutdown();
      return;
    }

    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHOK | ECHONL | ISIG);
    newt.c_iflag &= ~(IXON | IXOFF | ICRNL | INLCR);
    newt.c_cc[VMIN] = 0;
    newt.c_cc[VTIME] = 1;

    tcsetattr(input_fd, TCSANOW, &newt);
    tcflush(input_fd, TCIFLUSH);

    while (rclcpp::ok())
    {
      char c = 0;
      ssize_t n = read(input_fd, &c, 1);

      if (n == 1)
      {
        if (c == 'l' || c == 'L')
          //Forward a lidar
          publishNextLidar(1);
        else if (c == 'c' || c == 'C')
          //Forward a camera
          publishNextCamera(1);
        else if (c == 'p' || c == 'P')
          //Back a lidar
          publishNextLidar(-1);
        else if (c == 'b' || c == 'B')
          //Back a camera
          publishNextCamera(-1);
        else if (c == 'q' || c == 'Q')
          rclcpp::shutdown();
      }

      std::this_thread::sleep_for(50ms);
    }

    tcsetattr(input_fd, TCSANOW, &oldt);

    if (input_fd != STDIN_FILENO)
      close(input_fd);
  }

  void replayLoop()
  {
    while (rclcpp::ok())
    {
      if (events_.empty())
        return;

      //if(should_log) {RCLCPP_INFO(this->get_logger(), "Odometry event size is %d.", odometry_events_.size()); }

      const long long start_data_ns = events_.front().timestamp_ns;
      const auto start_wall = std::chrono::steady_clock::now();

      for (size_t i = 0; rclcpp::ok() && i < events_.size(); ++i)
      {
        const auto& ev = events_[i];

        const long long data_offset_ns = ev.timestamp_ns - start_data_ns;
        const long long scaled_wait_ns =
            static_cast<long long>(static_cast<double>(data_offset_ns) / replay_rate_);

        const auto target_wall =
            start_wall + std::chrono::nanoseconds(scaled_wait_ns);

        std::this_thread::sleep_until(target_wall);

        if (!rclcpp::ok())
          break;

        publishEvent(ev, "Replay");
      }

      if (!loop_)
        break;

      if(should_log) { RCLCPP_INFO(this->get_logger(), "Replay loop complete, restarting from beginning."); }
      std::this_thread::sleep_for(100ms);
    }

    rclcpp::shutdown();
  }

  void publishNextLidar(int step)
  {
    if (lidar_events_.empty())
    {
      RCLCPP_WARN(this->get_logger(), "No lidar events available.");
      return;
    }

    publishEvent(lidar_events_[next_lidar_i_], "Manual lidar");
    next_lidar_i_ = (next_lidar_i_ + step) % lidar_events_.size();
  }

  void publishNextCamera(int step)
  {
    if (image_events_.empty())
    {
      RCLCPP_WARN(this->get_logger(), "No camera events available.");
      return;
    }

    publishEvent(image_events_[next_image_i_], "Manual camera");
    next_image_i_ = (next_image_i_ + step) % image_events_.size();
  }

  bool loadPngToImageMsg(const fs::path& image_path, sensor_msgs::msg::Image& msg)
  {
    png_image image;
    std::memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, image_path.string().c_str()))
      return false;

    image.format = PNG_FORMAT_RGB;
    std::vector<png_byte> buffer(PNG_IMAGE_SIZE(image));

    if (!png_image_finish_read(&image, nullptr, buffer.data(), 0, nullptr))
    {
      png_image_free(&image);
      return false;
    }

    msg.height = image.height;
    msg.width = image.width;
    msg.encoding = "rgb8";
    msg.is_bigendian = false;
    msg.step = image.width * 3;
    msg.data.assign(buffer.begin(), buffer.end());

    png_image_free(&image);
    return true;
  }

  bool loadPcdToCloudMsg(const fs::path& pcd_path, sensor_msgs::msg::PointCloud2& msg, size_t& points)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path.string(), cloud) == -1)
      return false;

    if (cloud.points.empty())
      return false;

    pcl::toROSMsg(cloud, msg);
    points = cloud.points.size();
    return true;
  }

  sensor_msgs::msg::CameraInfo buildCameraInfo(const sensor_msgs::msg::Image& image_msg) const
  {
    sensor_msgs::msg::CameraInfo cam_info;
    cam_info.header = image_msg.header;
    cam_info.width = image_msg.width;
    cam_info.height = image_msg.height;

    const double fx = (fx_ > 0.0) ? fx_ : static_cast<double>(image_msg.width) / 2.0;
    const double fy = (fy_ > 0.0) ? fy_ : static_cast<double>(image_msg.height) / 2.0;
    const double cx = (cx_ > 0.0) ? cx_ : static_cast<double>(image_msg.width) / 2.0;
    const double cy = (cy_ > 0.0) ? cy_ : static_cast<double>(image_msg.height) / 2.0;

    cam_info.k = {fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0};

    cam_info.p = {fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0};

    return cam_info;
  }

  void publishEvent(const TimedEvent& ev, const char* mode_label)
  {
    const rclcpp::Time stamp(ev.timestamp_ns);

    if (ev.type == EventType::IMAGE)
    {
      sensor_msgs::msg::Image::SharedPtr image_msg(new sensor_msgs::msg::Image());

      if (!loadPngToImageMsg(ev.path, *image_msg))
      {
        RCLCPP_WARN(this->get_logger(), "Failed to load image: %s", ev.path.c_str());
        return;
      }

      image_msg->header.stamp = stamp;
      image_msg->header.frame_id = camera_optical_frame_id;

      auto cam_info = buildCameraInfo(*image_msg);

      image_pub_.publish(image_msg);
      //RCLCPP_INFO_STREAM(this->get_logger(), "sent image with s: " << image_msg->header.stamp.sec << " n: " << image_msg->header.stamp.nanosec);
      //image_publisher_->publish(std::move(image_msg));
      camera_info_publisher_->publish(cam_info);

      if(should_log) {
      RCLCPP_INFO(this->get_logger(),
                  "%s IMAGE: %s ts=%lld",
                  mode_label,
                  fs::path(ev.path).filename().c_str(),
                  ev.timestamp_ns);
      }
    }
    else if (ev.type == EventType::CLOUD)
    {
      sensor_msgs::msg::PointCloud2 cloud_msg;
      size_t point_count = 0;

      if (!loadPcdToCloudMsg(ev.path, cloud_msg, point_count))
      {
        RCLCPP_WARN(this->get_logger(), "Failed to load PCD: %s", ev.path.c_str());
        return;
      }

      cloud_msg.header.stamp = stamp;
      cloud_msg.header.frame_id = lidar_frame_id_;

      cloud_publisher_->publish(std::move(cloud_msg));
      lidar_timestamp_publisher_->publish(cloud_msg.header.stamp);
      //RCLCPP_INFO_STREAM(this->get_logger(), "sent lidar with s: " << cloud_msg.header.stamp.sec << " n: " << cloud_msg.header.stamp.nanosec);

      if(should_log) {
      RCLCPP_INFO(this->get_logger(),
                  "%s CLOUD: %s [%zu points] ts=%lld",
                  mode_label,
                  fs::path(ev.path).filename().c_str(),
                  point_count,
                  ev.timestamp_ns);
      }
    } else {
      if (!last_odometry_line_.empty()) {
        geometry_msgs::msg::TwistWithCovarianceStamped twist_msg;
        if (odometryMsgsToTwist(last_odometry_line_, ev.path, twist_msg)) {
          //twist_msg.header.stamp = stamp;
          twist_publisher_->publish(twist_msg);
        }
      }
      last_odometry_line_ = ev.path;
        
      if(should_log) {
        RCLCPP_INFO(this->get_logger(),
                "%s ODOMETRY: %s ts=%lld",
                mode_label,
                ev.path.c_str(),
                ev.timestamp_ns);
      }
    }
  }

  void parseOdometryMessage(const std::string& odometry_line, OdometryMessage& odometry_msg) {
    std::istringstream iss(odometry_line); // Create a string stream from the line

    std::string word; // Temporary string to hold extracted data

    std::getline(iss, word, ',');
    const long long nano_sec = std::stoll(word);

    odometry_msg.time = nano_sec;

    std::getline(iss, word, ',');
    odometry_msg.x = std::stod(word);

    std::getline(iss, word, ',');
    odometry_msg.y = std::stod(word);

    std::getline(iss, word, ',');
    odometry_msg.z = std::stod(word);

    std::getline(iss, word, ',');
    odometry_msg.qx = std::stod(word);

    std::getline(iss, word, ',');
    odometry_msg.qy = std::stod(word);

    std::getline(iss, word, ',');
    odometry_msg.qz = std::stod(word);

    std::getline(iss, word, ',');
    odometry_msg.qw = std::stod(word);
  }

  bool odometryMsgsToTwist(const std::string& last_odometry_line, const std::string& this_odometry_line, geometry_msgs::msg::TwistWithCovarianceStamped& twist_msg) {
    OdometryMessage last_odom;
    OdometryMessage curr_odom;

    parseOdometryMessage(last_odometry_line, last_odom);
    parseOdometryMessage(this_odometry_line, curr_odom);

    const rclcpp::Time currTime(curr_odom.time);
    const rclcpp::Time lastTime(last_odom.time);

    const rclcpp::Duration dt = currTime - lastTime;
    if (dt == rclcpp::Duration::from_seconds(0)) {
      return false;
    }

    //Estimate time of velocity to be between two odometry messages
    const rclcpp::Time time_stamp((lastTime.nanoseconds() + currTime.nanoseconds()) / 2);
    twist_msg.header.set__stamp(time_stamp);

    //Estimate linear velocity
    double dx = curr_odom.x - last_odom.x;
    double dy = curr_odom.y - last_odom.y;
    double dz = curr_odom.z - last_odom.z;

    auto linearVel = tf2::Vector3(dx, dy, dz) / (dt.seconds()); 

    twist_msg.twist.twist.linear.set__x(linearVel.getX());
    twist_msg.twist.twist.linear.set__y(linearVel.getY());
    twist_msg.twist.twist.linear.set__z(linearVel.getZ());

    //Estimate angular velocity
    auto lastQuaternion = tf2::Quaternion(last_odom.qw, last_odom.qx, last_odom.qy, last_odom.qz);
    auto currQuaternion = tf2::Quaternion(curr_odom.qw, curr_odom.qx, curr_odom.qy, curr_odom.qz);

    auto rotQuaternion = currQuaternion*lastQuaternion.inverse();

    double rollVel; 
    double pitchVel; 
    double yawVel; 

    tf2::Matrix3x3(rotQuaternion).getRPY(rollVel, pitchVel, yawVel);
    rollVel /= dt.seconds();
    pitchVel /= dt.seconds();
    yawVel /= dt.seconds();

    twist_msg.twist.twist.angular.set__x(rollVel);
    twist_msg.twist.twist.angular.set__y(pitchVel);
    twist_msg.twist.twist.angular.set__z(yawVel);
    return true;
  }

  void publishCameraOpticalTransform() {
    RCLCPP_INFO(this->get_logger(), "publishing transform");
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = camera_frame_id_;
    transform.child_frame_id = camera_optical_frame_id;
    transform.header.stamp = rclcpp::Clock().now();

    transform.transform.translation.x = 0;
    transform.transform.translation.y = 0;
    transform.transform.translation.z = 0;

    tf2::Quaternion rotation_quat;
    rotation_quat.setRPY(-M_PI / 2, 0, -M_PI / 2);
    transform.transform.rotation = tf2::toMsg(rotation_quat);

    static_tf_broadcaster_->sendTransform(transform);
  }
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(DatasetPublisher);
