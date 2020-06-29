#include "simulator.hpp"
#include "gjk.hpp"
using namespace racecar_simulator;
StandaloneSimulator::StandaloneSimulator(int num_cars, double timestep, double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass) {
    num_agents = num_cars;
    delt_t = timestep;
    map_exists = false;
    agent_poses.reserve(num_agents);
    for (int i=0; i<num_agents; i++) {
        if (i == ego_agent_idx) {
            RaceCar ego_car = RaceCar(delt_t, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, true);
            agents.push_back(ego_car);
        } else {
            RaceCar agent = RaceCar(delt_t, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, false);
            agents.push_back(agent);
        }
    }
}
StandaloneSimulator::~StandaloneSimulator() {
}
void StandaloneSimulator::set_map(std::vector<double> map, int map_height, int map_width, double map_resolution, double origin_x, double origin_y, double free_threshold) {
    this->map = map;
    this->map_height = map_height;
    this->map_width = map_width;
    this->map_resolution = map_resolution;
    this->origin_x = origin_x;
    this->origin_y = origin_y;
    this->free_threshold = free_threshold;
    for (RaceCar &agent : agents) {
        agent.set_map(map, map_height, map_width, map_resolution, origin_x, origin_y, free_threshold);
    }
    map_exists = true;
}
void StandaloneSimulator::update_params(double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass) {
    for (size_t i=0; i<agents.size(); i++) {
        agents[i].update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass);
    }
}
bool StandaloneSimulator::get_map_status() {
    return map_exists;
}
std::vector<CarObs> StandaloneSimulator::step(std::vector<double> velocities, std::vector<double> steering_angles) {
    if (!map_exists) {
    }
    for (size_t i=0; i<agents.size(); i++) {
        agents[i].set_velocity(velocities[i]);
        agents[i].set_steering_angle(steering_angles[i]);
    }
    for (size_t i=0; i<agents.size(); i++) {
        agents[i].update_pose();
        agent_poses[i] = agents[i].get_pose();
    }
    std::vector<Pose2D> ego_pose, op_pose;
    ego_pose.push_back(agent_poses[0]);
    op_pose.push_back(agent_poses[1]);
    agents[0].update_op_poses(op_pose);
    agents[1].update_op_poses(ego_pose);
    std::vector<CarObs> all_obs;
    for (size_t i=0; i<agents.size(); i++) {
        CarObs agent_obs = agents[i].update_scan();
        all_obs.push_back(agent_obs);
    }
    bool collision = check_collision();
    if (collision) {
        for (size_t i=0; i<agents.size(); i++) {
            all_obs[i].in_collision = true;
            all_obs[i].collision_angle = -100;
        }
    }
    current_obs.clear();
    current_obs = all_obs;
    return all_obs;
}
bool StandaloneSimulator::check_collision() {
    Pose2D ego_pose = agents[0].get_pose();
    Pose2D op_pose = agents[1].get_pose();
    double ego_x = ego_pose.x;
    double ego_y = ego_pose.y;
    double op_x = op_pose.x;
    double op_y = op_pose.y;
    if (sqrt(std::pow((ego_x-op_x),2) + std::pow((ego_y-op_y), 2)) > safety_radius) {
        return false;
    } else {
        Eigen::Matrix4d op_trans_mat = get_transformation_matrix(op_pose);
        Eigen::Matrix4d ego_trans_mat = get_transformation_matrix(ego_pose);
        Eigen::Vector4d rear_left_homo, rear_right_homo, front_left_homo, front_right_homo;
        rear_left_homo << -car_length/2, car_width/2, 0.0, 1.0;
        rear_right_homo << -car_length/2, -car_width/2, 0.0, 1.0;
        front_left_homo << car_length/2, car_width/2, 0.0, 1.0;
        front_right_homo << car_length/2, -car_width/2, 0.0, 1.0;
        Eigen::Vector4d ego_rear_left_transformed = ego_trans_mat*rear_left_homo;
        Eigen::Vector4d ego_rear_right_transformed = ego_trans_mat*rear_right_homo;
        Eigen::Vector4d ego_front_left_transformed = ego_trans_mat*front_left_homo;
        Eigen::Vector4d ego_front_right_transformed = ego_trans_mat*front_right_homo;
        ego_rear_left_transformed = ego_rear_left_transformed / ego_rear_left_transformed(3);
        ego_rear_right_transformed = ego_rear_right_transformed / ego_rear_right_transformed(3);
        ego_front_left_transformed = ego_front_left_transformed / ego_front_left_transformed(3);
        ego_front_right_transformed = ego_front_right_transformed / ego_front_right_transformed(3);
        Eigen::Vector4d op_rear_left_transformed = op_trans_mat*rear_left_homo;
        Eigen::Vector4d op_rear_right_transformed = op_trans_mat*rear_right_homo;
        Eigen::Vector4d op_front_left_transformed = op_trans_mat*front_left_homo;
        Eigen::Vector4d op_front_right_transformed = op_trans_mat*front_right_homo;
        op_rear_left_transformed = op_rear_left_transformed / op_rear_left_transformed(3);
        op_rear_right_transformed = op_rear_right_transformed / op_rear_right_transformed(3);
        op_front_left_transformed = op_front_left_transformed / op_front_left_transformed(3);
        op_front_right_transformed = op_front_right_transformed / op_front_right_transformed(3);
        vec2 ego_rl{-ego_rear_left_transformed(1), ego_rear_left_transformed(0)};
        vec2 ego_rr{-ego_rear_right_transformed(1), ego_rear_right_transformed(0)};
        vec2 ego_fl{-ego_front_left_transformed(1), ego_front_left_transformed(0)};
        vec2 ego_fr{-ego_front_right_transformed(1), ego_front_right_transformed(0)};
        std::vector<vec2> ego_vertices{ego_rl, ego_rr, ego_fr, ego_fl};
        vec2 op_rl{-op_rear_left_transformed(1), op_rear_left_transformed(0)};
        vec2 op_rr{-op_rear_right_transformed(1), op_rear_right_transformed(0)};
        vec2 op_fl{-op_front_left_transformed(1), op_front_left_transformed(0)};
        vec2 op_fr{-op_front_right_transformed(1), op_front_right_transformed(0)};
        std::vector<vec2> op_vertices{op_fl, op_rl, op_rr, op_fr};
        int collision = gjk(op_vertices, ego_vertices);
        return static_cast<bool>(collision);
    }
}
void StandaloneSimulator::reset() {
    current_obs.clear();
    for (RaceCar &agent : agents) {
        agent.reset();
    }
}
void StandaloneSimulator::reset_bypose(std::vector<Pose2D> &poses) {
    current_obs.clear();
    for (size_t i=0; i<poses.size(); i++) {
        agents[i].reset_bypose(poses[i]);
    }
}
Eigen::Matrix4d StandaloneSimulator::get_transformation_matrix(const Pose2D &pose) {
    double x = pose.x;
    double y = pose.y;
    double theta = pose.theta;
    double cosine = std::cos(theta);
    double sine = std::sin(theta);
    Eigen::Matrix4d T;
    T << cosine, -sine, 0.0, x, sine, cosine, 0.0, y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return T;
}
