#pragma once         
namespace racecar_simulator {
struct CarState {
    double x;
    double y;
    double theta;
    double velocity;
    double steer_angle;
    double angular_velocity;
    double slip_angle;
    bool st_dyn;
};
}
