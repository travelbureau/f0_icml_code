#include "pose_2d.hpp"
#include "scan_simulator_2d.hpp"
#include "distance_transform.hpp"
using namespace racecar_simulator;
ScanSimulator2D::ScanSimulator2D(
    int num_beams_,
    double field_of_view_,
    double scan_std_dev_,
    double ray_tracing_epsilon_,
    int theta_discretization)
  : num_beams(num_beams_),
    field_of_view(field_of_view_),
    scan_std_dev(scan_std_dev_),
    ray_tracing_epsilon(ray_tracing_epsilon_),
    theta_discretization(theta_discretization)
{
  angle_increment = field_of_view/(num_beams - 1);
  scan_output = std::vector<double>(num_beams);
  noise_generator = std::mt19937(12345);
  noise_dist = std::normal_distribution<double>(0., scan_std_dev);
  theta_index_increment = theta_discretization * angle_increment/(2 * M_PI);
  sines = std::vector<double>(theta_discretization + 1);
  cosines = std::vector<double>(theta_discretization + 1);
  for (int i = 0; i <= theta_discretization; i++) {
    double theta = (2 * M_PI * i)/((double) theta_discretization);
    sines[i] = std::sin(theta);
    cosines[i] = std::cos(theta);
  }
}
const std::vector<double> ScanSimulator2D::scan(const Pose2D & pose) {
  scan(pose, scan_output.data());
  return scan_output;
}
void ScanSimulator2D::scan(const Pose2D & pose, double * scan_data) {
  double theta_index =
    theta_discretization * (pose.theta - field_of_view/2.)/(2 * M_PI);
  theta_index = std::fmod(theta_index, theta_discretization);
  while (theta_index < 0) theta_index += theta_discretization;
  for (int i = 0; i < num_beams; i++) {
    scan_data[i] = trace_ray(pose.x, pose.y, theta_index);
    if (scan_std_dev > 0)
        scan_data[i] += noise_dist(noise_generator);
    theta_index += theta_index_increment;
    while (theta_index >= theta_discretization)
      theta_index -= theta_discretization;
  }
}
double ScanSimulator2D::trace_ray(double x, double y, double theta_index) const {
  int theta_index_ = theta_index + 0.5;
  double s = sines[theta_index_];
  double c = cosines[theta_index_];
  double distance_to_nearest = distance_transform(x, y);
  double total_distance = distance_to_nearest;
  while (distance_to_nearest > ray_tracing_epsilon) {
    x += distance_to_nearest * c;
    y += distance_to_nearest * s;
    distance_to_nearest = distance_transform(x, y);
    total_distance += distance_to_nearest;
  }
  return total_distance;
}
double ScanSimulator2D::distance_transform(double x, double y) const {
  int cell = xy_to_cell(x, y);
  if (cell < 0) return 0;
  return dt[cell];
}
void ScanSimulator2D::set_map(
    const std::vector<double> & map,
    size_t height_,
    size_t width_,
    double resolution_,
    const Pose2D & origin_,
    double free_threshold) {
  height = height_;
  width = width_;
  resolution = resolution_;
  origin = origin_;
  origin_c = std::cos(origin.theta);
  origin_s = std::sin(origin.theta);
  dt = std::vector<double>(map.size());
  for (size_t i = 0; i < map.size(); i++) {
    if (0 <= map[i] and map[i] <= free_threshold) {
      dt[i] = 99999;
    } else {
      dt[i] = 0;
    }
  }
  DistanceTransform::distance_2d(dt, width, height, resolution);
}
void ScanSimulator2D::set_map(const std::vector<double> & map, double free_threshold) {
  for (size_t i = 0; i < map.size(); i++) {
    if (0 <= map[i] and map[i] <= free_threshold) {
      dt[i] = 99999;
    } else {
      dt[i] = 0;
    }
  }
  DistanceTransform::distance_2d(dt, width, height, resolution);
}
void ScanSimulator2D::xy_to_row_col(double x, double y, int * row, int * col) const {
  double x_trans = x - origin.x;
  double y_trans = y - origin.y;
  double x_rot = x_trans * origin_c + y_trans * origin_s;
  double y_rot = - x_trans * origin_s + y_trans * origin_c;
  if (x_rot < 0 or x_rot >= width * resolution or
      y_rot < 0 or y_rot >= height * resolution) {
    *col = -1;
    *row = -1;
  } else {
    *col = std::floor(x_rot/resolution);
    *row = std::floor(y_rot/resolution);
  }
}
int ScanSimulator2D::row_col_to_cell(int row, int col) const {
  return row * width + col;
}
int ScanSimulator2D::xy_to_cell(double x, double y) const {
  int row, col;
  xy_to_row_col(x, y, &row, &col);
  return row_col_to_cell(row, col);
}
