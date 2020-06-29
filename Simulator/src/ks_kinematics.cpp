#include <cmath>
#include "car_state.hpp"
#include "ks_kinematics.hpp"
using namespace racecar_simulator;
CarState KSKinematics::update(
        const CarState start,
        double accel,
        double steer_angle_vel,
        CarParams p,
        double dt) {
    CarState end;
    double x_dot = start.velocity * std::cos(start.theta);
    double y_dot = start.velocity * std::sin(start.theta);
    double v_dot = accel;
    double steer_ang_dot = steer_angle_vel;
    double theta_dot = start.velocity / p.wheelbase * std::tan(start.steer_angle);
    double friction_term = 0;
    if (start.velocity > 0) {
        friction_term = -p.friction_coeff;
    } else if (start.velocity < 0) {
        friction_term = p.friction_coeff;
    }
    double fr_factor = .1;
    v_dot += fr_factor * friction_term;
    end.x = start.x + x_dot * dt;
    end.y = start.y + y_dot * dt;
    end.theta = start.theta + theta_dot * dt;
    end.velocity = start.velocity + v_dot * dt;
    end.steer_angle = start.steer_angle + steer_ang_dot * dt;
    end.angular_velocity = start.angular_velocity;
    end.slip_angle = start.slip_angle;
    return end;
}
