cmake_minimum_required(VERSION 3.8)
project(tracer_a1_qpmc_adapter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

set(A1_QP_MPC_ROSFREE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/a1_qp_mpc_rosfree
)

add_library(a1_qp_mpc_rosfree
  ${A1_QP_MPC_ROSFREE_DIR}/utils/Utils.cpp
  ${A1_QP_MPC_ROSFREE_DIR}/legKinematics/A1Kinematics.cpp
  ${A1_QP_MPC_ROSFREE_DIR}/A1RobotControl.cpp
  ${A1_QP_MPC_ROSFREE_DIR}/ConvexMpc.cpp
)

target_include_directories(a1_qp_mpc_rosfree PUBLIC
  ${A1_QP_MPC_ROSFREE_DIR}
  ${A1_QP_MPC_ROSFREE_DIR}/utils
  ${A1_QP_MPC_ROSFREE_DIR}/legKinematics
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(a1_qp_mpc_rosfree
  Eigen3::Eigen
  OsqpEigen::OsqpEigen
)

add_executable(tracer_a1_qpmc_adapter_node
  src/tracer_a1_qpmc_adapter_node.cpp
)

ament_target_dependencies(tracer_a1_qpmc_adapter_node
  rclcpp
  std_msgs
)

target_link_libraries(tracer_a1_qpmc_adapter_node
  a1_qp_mpc_rosfree
)

install(TARGETS tracer_a1_qpmc_adapter_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
