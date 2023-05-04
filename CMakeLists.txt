cmake_minimum_required(VERSION 3.10)

project(scara_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(lcm)
find_package(Eigen3)

include_directories(third_party/mujoco-2.3.5/include)
include_directories(third_party/mujoco-2.3.5/simulate)
include_directories(msgs/cpp)

include_directories(third_party/lodepng)

file(GLOB LODEPNG_SRC third_party/lodepng/*.cpp)
add_library(lodepng ${LODEPNG_SRC})

link_directories(third_party/mujoco-2.3.5/lib)

file(GLOB MUJOCO_SIM_SRC third_party/mujoco-2.3.5/simulate/*.cc)

add_executable(${PROJECT_NAME} ${MUJOCO_SIM_SRC})
target_link_libraries(${PROJECT_NAME} mujoco glfw pthread lcm lodepng glog)

