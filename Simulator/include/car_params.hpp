#pragma once         
namespace racecar_simulator {
struct CarParams {
    double wheelbase;
    double friction_coeff;
    double h_cg;
    double l_f;
    double l_r;
    double cs_f;
    double cs_r;
    double mass;
    double I_z;
};
}
