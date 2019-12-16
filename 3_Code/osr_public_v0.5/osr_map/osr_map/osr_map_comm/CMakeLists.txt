cmake_minimum_required(VERSION 2.8.3)
project(osr_map_comm)

add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-O2")


find_package(catkin REQUIRED COMPONENTS
  # grid_map
  grid_map_core

)

find_package(Boost REQUIRED COMPONENTS system thread serialization)

find_package(Eigen3 QUIET)




catkin_package(CATKIN_DEPENDS
               DEPENDS Boost #Eigen3
               INCLUDE_DIRS include ${BOOST_INCLUDE_DIRS} #${EIGEN3_INCLUDE_DIR}
               LIBRARIES ${PROJECT_NAME})
###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${BOOST_INCLUDE_DIRS}
  #${EIGEN3_INCLUDE_DIR}
)

# Declare a cpp library
add_library(${PROJECT_NAME}
  src/comp_comm.cpp
  src/comp_converter.cpp
  src/comp_packets.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}  ${Boost_LIBRARIES} 
  # ${EIGEN3_LIBRARIES}
)



#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
#  PATTERN ".svn" EXCLUDE
)

# Mark other files for installation
#install(
#  DIRECTORY doc
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)
