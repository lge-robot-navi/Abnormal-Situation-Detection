cmake_minimum_required(VERSION 2.8.3)
project(osr_demos)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  # tf2_ros
  # std_msgs
  # nav_msgs
  # geometry_msgs
  # sensor_msgs
  # image_transport
  # message_generation
  
  
  # osr
  osr_comm
  osr_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
catkin_package(
  INCLUDE_DIRS 
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
  DEPENDS
)

# change_image_sender
add_executable(change_image_sender
  src/change_image_sender.cpp
)
target_link_libraries(change_image_sender ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# change_image_receiver
add_executable(change_image_receiver
  src/change_image_receiver.cpp
)
target_link_libraries(change_image_receiver ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# change_image_sender_multiport
add_executable(change_image_sender_multiport
  src/change_image_sender_multiport.cpp
)
target_link_libraries(change_image_sender_multiport ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# change_image_receiver_multiport
add_executable(change_image_receiver_multiport
  src/change_image_receiver_multiport.cpp
)
target_link_libraries(change_image_receiver_multiport ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# abnormals_sender
add_executable(abnormals_sender
  src/abnormals_sender.cpp
)
target_link_libraries(abnormals_sender ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# abnormals_sender_multiport
add_executable(abnormals_sender_multiport
  src/abnormals_sender_multiport.cpp
)
target_link_libraries(abnormals_sender_multiport ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# abnormals_receiver
add_executable(abnormals_receiver
  src/abnormals_receiver.cpp
)
target_link_libraries(abnormals_receiver ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# abnormals_receiver_multiport
add_executable(abnormals_receiver_multiport
  src/abnormals_receiver_multiport.cpp
)
target_link_libraries(abnormals_receiver_multiport ${catkin_LIBRARIES} ${Boost_LIBRARIES} )
