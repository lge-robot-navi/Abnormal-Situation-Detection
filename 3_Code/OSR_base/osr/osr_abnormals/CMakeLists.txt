cmake_minimum_required(VERSION 2.8.3)
project(osr_abnormals)

add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-O2")

find_package(catkin REQUIRED COMPONENTS

  roscpp
  osr_msgs

  # grid_map
  grid_map_core
  # grid_map_cv
  
  # ros dependency for viz
  grid_map_ros
  grid_map_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
link_directories(${catkin_LIBRARY_DIRS})

# user_defined_abnormal
add_executable(user_defined_abnormal_node
  src/user_defined_abnormal_node.cpp
  src/user_defined_abnormal.cpp
)
target_link_libraries(user_defined_abnormal_node ${catkin_LIBRARIES})

# agent_info_viewer
add_executable(agent_info_viewer
  src/agent_info_viewer_node.cpp
  src/agent_info_viewer.cpp
)
target_link_libraries(agent_info_viewer ${catkin_LIBRARIES})