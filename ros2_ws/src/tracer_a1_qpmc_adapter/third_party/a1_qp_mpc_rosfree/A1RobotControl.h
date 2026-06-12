//
// Created by shuoy on 10/19/21.
//

#ifndef A1_CPP_A1ROBOTCONTROL_H
#define A1_CPP_A1ROBOTCONTROL_H

#include <iostream>
#include <string>
#include <chrono>

// to debug

// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>

#include "A1Params.h"
#include "A1CtrlStates.h"
#include "utils/Utils.h"
#include "ConvexMpc.h"

#include "utils/filter.hpp"


class A1RobotControl {
public:
    A1RobotControl();

    void update_plan(A1CtrlStates &state, double dt);

    void generate_swing_legs_ctrl(A1CtrlStates &state, double dt);

    void compute_joint_torques(A1CtrlStates &state);

    Eigen::Matrix<double, 3, NUM_LEG> compute_grf(A1CtrlStates &state, double dt);

    Eigen::Vector3d compute_walking_surface(A1CtrlStates &state);

private:
    BezierUtils bezierUtils[NUM_LEG];

    Eigen::Matrix<double, 6, 1> root_acc;
    // allocate the problem weight matrices
    Eigen::DiagonalMatrix<double, 6> Q;
    double R;
    // ground friction coefficient
    double mu;
    double F_min;
    double F_max;
    // allocate QP problem matrices and vectors
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd gradient;
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;


    OsqpEigen::Solver solver;

    // ROS debug publishers removed for ROS2/TRACER adapter.

    //MPC does not start for the first 10 ticks to prevent uninitialized NAN goes into joint_torques
    int mpc_init_counter;

    std::string use_sim_time;

    // filters
    MovingWindowFilter terrain_angle_filter;
    MovingWindowFilter recent_contact_x_filter[NUM_LEG];
    MovingWindowFilter recent_contact_y_filter[NUM_LEG];
    MovingWindowFilter recent_contact_z_filter[NUM_LEG];
};


#endif //A1_CPP_A1ROBOTCONTROL_H
