cmake_minimum_required(VERSION 2.8.3)
project(osr_map_agents)

add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-O2")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  osr_msgs
  image_transport
  # message_generation
  cv_bridge
  # tf2_ros
  
  # grid_map
  grid_map_core
  grid_map_msgs
  grid_map_ros
  grid_map_cv
  
  # pcl
  pcl_conversions
  pcl_ros
  
  # osr_map
  osr_map_comm
)
find_package(cmake_modules REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(PCL REQUIRED)
# find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  # ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${catkin_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
  	image_transport
    cv_bridge
  DEPENDS
    #Eigen3
)

# elevation_layer_node
add_executable(fixed_elevation_layer_node
  src/map_layers/fixed_elevation_layer.cpp
  src/fixed_elevation_layer_node.cpp
)
target_link_libraries(fixed_elevation_layer_node ${catkin_LIBRARIES})

# elevation_layer_node
add_executable(elevation_layer_node
  src/map_layers/elevation_layer.cpp
  src/elevation_layer_node.cpp
)
target_link_libraries(elevation_layer_node ${catkin_LIBRARIES})

# object_layer_node
add_executable(object_layer_node
  src/map_layers/object_layer.cpp
  src/object_layer_node.cpp
)
target_link_libraries(object_layer_node ${catkin_LIBRARIES})
# object_layer_node
add_executable(fixed_object_layer_node
  src/map_layers/fixed_object_layer.cpp
  src/fixed_object_layer_node.cpp
  src/se3.cpp
  src/virtual_camera.cpp
)
target_link_libraries(fixed_object_layer_node ${catkin_LIBRARIES} ${OpenCV_LIBS})

# thermal_layer_node
add_executable(thermal_layer_node
  src/map_layers/thermal_layer.cpp
  src/thermal_layer_node.cpp
)
target_link_libraries(thermal_layer_node ${catkin_LIBRARIES})

# osr_map_agent_node
add_executable(osr_map_agent_node
  src/osr_map_agent_node.cpp src/agent.cpp
)
target_link_libraries(osr_map_agent_node ${catkin_LIBRARIES})

# osr_map_agent_multiport_node
add_executable(osr_map_agent_multiport_node
  src/osr_map_agent_multiport_node.cpp src/agent.cpp
)
target_link_libraries(osr_map_agent_multiport_node ${catkin_LIBRARIES})

