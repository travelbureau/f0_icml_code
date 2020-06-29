#pragma once         
namespace racecar_simulator {
struct CarOdom {
    double x;
    double y;
    double z;
    double qx;
    double qy;
    double qz;
    double qw;
    double linear_x;
    double linear_y;
    double linear_z;
    double angular_x;
    double angular_y;
    double angular_z;
};
}
