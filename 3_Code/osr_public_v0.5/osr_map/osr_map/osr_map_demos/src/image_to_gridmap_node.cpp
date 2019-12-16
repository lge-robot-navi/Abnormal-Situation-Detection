/**
 * @file		image_to_gridmap_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running image_to_gridmap_node
 * 				to visualize RGB image as grid map through occupancy grid map and point cloud on RVIZ
 * @remark
 * @warning
 *
 * Copyright (C) 2019  <Kiin Na>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */




#include <ros/ros.h>
#include "osr_map_demos/image_to_gridmap.h"

int main(int argc, char** argv)
{
  // Initialize node and publisher.
  ros::init(argc, argv, "image_to_gridmap_node");
  ros::NodeHandle nh("~");
  osr_map::image_to_gridmap img_to_map(nh);

  while(ros::ok()){
	img_to_map.run();
  }
  return 0;
}

