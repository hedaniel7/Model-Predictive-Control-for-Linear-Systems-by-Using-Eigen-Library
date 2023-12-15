cmake_minimum_required(VERSION 3.26)
project(Model_Predictive_Control_for_Linear_Systems_by_Using_Eigen_Library)

set(CMAKE_CXX_STANDARD 14)

#find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)

# Manually specify the Eigen include directory
#set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")

include_directories(${CMAKE_SOURCE_DIR}/eigen-3.4.0)


#include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(MPC
        driver_code.cpp
        ModelPredictiveController.cpp
        ModelPredictiveController.h)

#target_link_libraries(MPC Eigen3::Eigen)

#set_target_properties(MPC PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
