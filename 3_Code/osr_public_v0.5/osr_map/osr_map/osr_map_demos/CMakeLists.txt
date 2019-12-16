cmake_minimum_required(VERSION 2.8.3)
project(osr_map_demos)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  
  # grid_map
  grid_map_core
  grid_map_cv
  
  # ros dependency for viz
  grid_map_ros
  grid_map_msgs
)
#find_package(cmake_modules REQUIRED)
#find_package(Eigen3 REQUIRED)
#find_package(TinyXML REQUIRED)
#find_package(Boost REQUIRED COMPONENTS system thread serialization)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
# ${EIGEN3_INCLUDE_DIR}
# ${PCL_INCLUDE_DIRS}
# ${TinyXML_INCLUDE_DIRS}
# ${BOOST_INCLUDE_DIRS}
)
link_directories(${catkin_LIBRARY_DIRS})

# osr_map_server_node
add_executable(image_to_gridmap
  src/image_to_gridmap.cpp
  src/image_to_gridmap_node.cpp
)
target_link_libraries(image_to_gridmap ${catkin_LIBRARIES})
