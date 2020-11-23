/**
 * @file		thermal_layer_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running thermal_layer_node
 * 				to build thermal layer of mobile agent
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


#include "osr_map_agents/map_layers/thermal_layer.h"

int main(int argc, char **argv)
{
  	// Node Generation
  	ros::init(argc, argv, "thermal_layer");

  	ros::NodeHandle nh("~");

  	std::string robot_frame_id, map_frame_id, world_frame_id;
  	std::string layer_topic, pose_topic, thermal_image_topic, pointcloud_topic, thermal_to_lidar_info_topic;
  	float length_x, length_y, resolution;
  	float min_thermal, max_thermal, min_range, max_range;
  	float period;

  	int robot_id;
  	nh.param<int>("robot_id", robot_id, 1);


  	nh.param<float>("period", period, 10);
  	nh.param<float>("length_x", length_x, 30);
  	nh.param<float>("length_y", length_y, 30);
  	nh.param<float>("resolution", resolution, 0.5);

  	nh.param<float>("min_thermal", min_thermal, 0.0);
  	nh.param<float>("max_thermal", max_thermal, 100.0);
  	nh.param<float>("min_range", min_range, 0.5);
  	nh.param<float>("max_range", max_range, 30.0);

  	nh.param<std::string>("layer_topic", layer_topic, "/thermal_layer");
  	nh.param<std::string>("pose_topic", pose_topic, "/robot_odom");
  	nh.param<std::string>("pointcloud_topic", pointcloud_topic, "/osr/lidar_pointcloud");
  	nh.param<std::string>("thermal_image_topic", thermal_image_topic, "/osr/image_thermal");
  	nh.param<std::string>("thermal_to_lidar_info_topic", thermal_to_lidar_info_topic, "/osr/image_thermal/camera_info");

  	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
  	nh.param<std::string>("map_frame_id", map_frame_id, "map");
  	nh.param<std::string>("world_frame_id", world_frame_id, "world");

  	osr_map::thermal_layer thermal_layer_node(nh, period, robot_id, robot_frame_id, map_frame_id);
  	thermal_layer_node.set_comm(layer_topic, pose_topic, thermal_image_topic, pointcloud_topic, thermal_to_lidar_info_topic);
  	thermal_layer_node.set_map_param(length_x, length_y, resolution);
  	thermal_layer_node.set_thermal_param(min_thermal, max_thermal);
  	thermal_layer_node.set_range_param(min_range, max_range);

  	while (ros::ok())
		thermal_layer_node.run();

	return 0;
}
