/**
 * @file		object_layer_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running object_layer_node
 * 				to build object_layer of mobile agent
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

#include "osr_map_agents/map_layers/object_layer.h"

int main(int argc, char **argv)
{
  	// Node Generation
  	ros::init(argc, argv, "object_layer");


  	ros::NodeHandle nh("~");

  	std::string robot_frame_id, map_frame_id, world_frame_id;
  	std::string layer_topic, object_topic;
  	float length_x, length_y, resolution;
  	float period;
    bool multi_project_on;
  	int robot_id;
  	nh.param<int>("robot_id", robot_id, 1);

  	nh.param<float>("period", period, 10);
  	nh.param<float>("length_x", length_x, 50);
  	nh.param<float>("length_y", length_y, 50);
  	nh.param<float>("resolution", resolution, 0.5);

  	nh.param<bool>("multi_project_on", multi_project_on, true);

  	nh.param<std::string>("layer_topic", layer_topic, "/object_layer");
  	nh.param<std::string>("object_topic", object_topic, "/tracks");

  	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
  	nh.param<std::string>("map_frame_id", map_frame_id, "osr_map");
  	nh.param<std::string>("world_frame_id", world_frame_id, "world");

  	float min_theta, max_theta, min_range, max_range;
  	nh.param<float>("min_theta", min_theta, -45);
	nh.param<float>("max_theta", max_theta, 45);
	nh.param<float>("min_range", min_range, 1.0);
	nh.param<float>("max_range", max_range, 15.0);


  	osr_map::object_layer object_layer_node(nh, period, robot_id, robot_frame_id, map_frame_id);
  	object_layer_node.set_comm(layer_topic, object_topic);
  	object_layer_node.set_map_param(length_x, length_y, resolution, multi_project_on);
  	object_layer_node.set_update_param(min_theta, max_theta, min_range, max_range);

  	while (ros::ok())
		object_layer_node.run();

	return 0;
}

