/**
 * @file		elevation_layer_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running elevation_layer_node
 * 				to build elevation layer of mobile agent
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


#include "osr_map_agents/map_layers/elevation_layer.h"

int main(int argc, char **argv)
{
  	// Node Generation
  	ros::init(argc, argv, "elevation_layer");

  	ros::NodeHandle nh("~");

  	std::string robot_frame_id, map_frame_id, world_frame_id;
  	std::string layer_topic, pose_topic, pointcloud_topic;
  	float length_x, length_y, resolution;
  	float offset_z, min_z, max_z, thickness;
  	float period;
  	float min_range, max_range;

  	int robot_id;
  	nh.param<int>("robot_id", robot_id, 1);


  	nh.param<float>("period", period, 10);
  	nh.param<float>("length_x", length_x, 50);
  	nh.param<float>("length_y", length_y, 50);
  	nh.param<float>("resolution", resolution, 0.5);

  	nh.param<float>("offset_z", offset_z, 0.0);
  	nh.param<float>("min_z", min_z, -0.5);
  	nh.param<float>("max_z", max_z, 2.0);
  	nh.param<float>("thickness", thickness, 0.3);
  	nh.param<float>("min_range", min_range, 1.0);
  	nh.param<float>("max_range", max_range, 20.0);

  	nh.param<std::string>("layer_topic", layer_topic, "/elevation_layer");
  	nh.param<std::string>("pose_topic", pose_topic, "/amcl_pose");
  	nh.param<std::string>("pointcloud_topic", pointcloud_topic, "/velodyne_points");

  	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
  	nh.param<std::string>("map_frame_id", map_frame_id, "osr_map");
  	nh.param<std::string>("world_frame_id", world_frame_id, "world");

  	osr_map::elevation_layer elevation_layer_node(nh, period, robot_id, robot_frame_id, map_frame_id);
  	elevation_layer_node.set_comm(layer_topic, pose_topic, pointcloud_topic);
  	elevation_layer_node.set_map_param(length_x, length_y, resolution);
  	elevation_layer_node.set_z_param(offset_z, min_z, max_z, thickness);
  	elevation_layer_node.set_range_param(min_range, max_range);

  	while (ros::ok())
		elevation_layer_node.run();

	return 0;
}
