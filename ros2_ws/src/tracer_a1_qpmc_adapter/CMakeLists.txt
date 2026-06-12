cmake_minimum_required(VERSION 3.8)
project(tracer_a1_qpmc_adapter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(tracer_a1_qpmc_adapter_node
  src/tracer_a1_qpmc_adapter_node.cpp
)

ament_target_dependencies(tracer_a1_qpmc_adapter_node
  rclcpp
  std_msgs
)

install(TARGETS tracer_a1_qpmc_adapter_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
