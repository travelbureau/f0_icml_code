#include <cstddef>
#include <vector>
#include <cmath>
#include "distance_transform.hpp"
using namespace racecar_simulator;
void DistanceTransform::distance_squared_1d(
    const std::vector<double> & input,
    std::vector<double> & output) {
  std::vector<size_t> parabola_idxs(input.size());
  parabola_idxs[0] = 0;
  std::vector<double> parabola_boundaries(input.size() + 1);
  parabola_boundaries[0] = -inf;
  parabola_boundaries[1] = inf;
  int num_parabolas = 0;
  double intersection_point;
  for (size_t idx = 1; idx < input.size(); idx++) {
    num_parabolas++;
    do {
      num_parabolas--;
      int parabola_idx = parabola_idxs[num_parabolas];
      intersection_point = (
              (input[idx] + idx * idx)
              -
              (input[parabola_idx] + parabola_idx * parabola_idx))
              /
              (2 * (idx - parabola_idx));
    } while (intersection_point <= parabola_boundaries[num_parabolas]);
    num_parabolas ++;
    parabola_idxs[num_parabolas] = idx;
    parabola_boundaries[num_parabolas] = intersection_point;
    parabola_boundaries[num_parabolas+1] = inf;
  }
  int parabola = 0;
  for (size_t idx = 0; idx < input.size(); idx++) {
    while (parabola_boundaries[parabola + 1] < idx) parabola++;
    int idx_dist = idx - parabola_idxs[parabola];
    output[idx] = idx_dist * idx_dist + input[parabola_idxs[parabola]];
  }
}
void DistanceTransform::distance_squared_2d(
    std::vector<double> & input,
    size_t width,
    size_t height,
    double boundary_value) {
  std::vector<double> col_vec(height + 2);
  std::vector<double> col_dt(height + 2);
  for (size_t col = 0; col < width; col++) {
    col_vec[0] = boundary_value;
    col_vec[height + 1] = boundary_value;
    for (size_t row = 0; row < height; row++) {
      col_vec[row + 1] = input[row * width + col];
    }
    distance_squared_1d(col_vec, col_dt);
    for (size_t row = 0; row < height; row++) {
      input[row * width + col] = col_dt[row + 1];
    }
  }
  std::vector<double> row_vec(width + 2);
  std::vector<double> row_dt(width + 2);
  for (size_t row = 0; row < height; row++) {
    row_vec[0] = boundary_value;
    row_vec[width + 1] = boundary_value;
    for (size_t col = 0; col < width; col++) {
      row_vec[col + 1] = input[row * width + col];
    }
    distance_squared_1d(row_vec, row_dt);
    for (size_t col = 0; col < width; col++) {
      input[row * width + col] = row_dt[col + 1];
    }
  }
}
void DistanceTransform::distance_2d(
    std::vector<double> & input,
    size_t width,
    size_t height,
    double resolution,
    double boundary_value) {
  distance_squared_2d(input, width, height, boundary_value);
  for (size_t i = 0; i < input.size(); i++) {
    input[i] = resolution * sqrt(input[i]);
  }
}
