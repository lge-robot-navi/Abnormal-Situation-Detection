cmake_minimum_required(VERSION 2.8.3)
project(osr_map_server)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-O2")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  # std_msgs
  # nav_msgs
  # geometry_msgs
  # sensor_msgs
  osr_msgs
  # message_generation
  cv_bridge
  # tf2_ros
  
  # grid_map
  grid_map_core
  grid_map_cv
  
  osr_map_comm
  
  # ros dependency for viz
  grid_map_ros
  grid_map_msgs
)
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TinyXML REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread serialization)

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
  ${PCL_INCLUDE_DIRS}
  ${TinyXML_INCLUDE_DIRS}
  ${BOOST_INCLUDE_DIRS}
)
link_directories(${catkin_LIBRARY_DIRS})

# short_term_server_node
add_executable(short_term_server_node
  src/short_term_server_node.cpp
  src/map_manager.cpp
)
target_link_libraries(short_term_server_node ${catkin_LIBRARIES} ${TinyXML_LIBRARIES}
)



