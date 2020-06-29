#pragma once         
#include "car_odom.hpp"
#include "car_state.hpp"
#include "pose_2d.hpp"
#include <vector>
namespace racecar_simulator {
struct CarObs {
 CarOdom odom;
    Pose2D pose;
 std::vector<double> scan;
 bool in_collision;
 double collision_angle;
};
}
