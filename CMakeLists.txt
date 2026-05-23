cmake_minimum_required(VERSION 3.20)
project(pose_trainer_blender LANGUAGES CXX)

include(FetchContent)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.4 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  )
  FetchContent_MakeAvailable(eigen)
endif()

pybind11_add_module(_pose_trainer_core
  src/pose_trainer_core/cpp/bindings.cpp
  src/pose_trainer_core/cpp/opencl_backend.cpp
  src/pose_trainer_core/cpp/pose_trainer_core.cpp
)

target_compile_features(_pose_trainer_core PRIVATE cxx_std_17)
target_include_directories(_pose_trainer_core PRIVATE src/pose_trainer_core/cpp)
if(TARGET Eigen3::Eigen)
  target_link_libraries(_pose_trainer_core PRIVATE Eigen3::Eigen)
else()
  target_include_directories(_pose_trainer_core PRIVATE ${eigen_SOURCE_DIR})
endif()

install(TARGETS _pose_trainer_core DESTINATION pose_trainer_core)
