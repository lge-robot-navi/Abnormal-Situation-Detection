/**
 * @file		fixed_object_layer_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na
 * @brief   	This file includes main function for running fixed_object_layer_node
 * 				to build object layer of fixed agent
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


#include "osr_map_agents/map_layers/fixed_object_layer.h"

int main(int argc, char **argv)
{
  	// Node Generation
  	ros::init(argc, argv, "fixed_object_layer");


  	ros::NodeHandle nh("~");

  	std::string robot_frame_id, map_frame_id, world_frame_id;
  	std::string pose_topic, layer_topic, object_topic;
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

  	nh.param<std::string>("pose_topic", pose_topic, "/robot_odom");
  	nh.param<std::string>("layer_topic", layer_topic, "/object_layer");
  	nh.param<std::string>("object_topic", object_topic, "/tracks");

  	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
  	nh.param<std::string>("map_frame_id", map_frame_id, "osr_map");
  	nh.param<std::string>("world_frame_id", world_frame_id, "world");

  	float camera_fx, camera_fy, camera_cx, camera_cy, camera_x, camera_y, camera_z, camera_rx, camera_ry, camera_rz;
  	nh.param<float>("camera_fx",     camera_fx, 612.4);
  	nh.param<float>("camera_fy",     camera_fy, 612.4);
  	nh.param<float>("camera_cx",     camera_cx, 320.0);
  	nh.param<float>("camera_cy",     camera_cy, 240.0);
  	nh.param<float>("camera_x", camera_x, 3.200);
  	nh.param<float>("camera_y", camera_y, 3.580);
  	nh.param<float>("camera_z", camera_z, 2.390);
	nh.param<float>("camera_pan",    camera_rx, -140.0);
	nh.param<float>("camera_tilt",   camera_ry, -23.6);
	nh.param<float>("camera_roll",   camera_rz, -1.0);

	double image_from_width, image_to_width, image_from_height, image_to_height;
	nh.param<double>("image_from_width",    image_from_width, 0);
	nh.param<double>("image_to_width",   image_to_width, 250);
	nh.param<double>("image_from_height",   image_from_height, 480);
	nh.param<double>("image_to_height",   image_to_height, 640);

  	osr_map::fixed_object_layer object_layer_node(nh, period, robot_id, robot_frame_id, map_frame_id);
  	object_layer_node.set_comm(pose_topic, layer_topic, object_topic);
  	object_layer_node.set_map_param(length_x, length_y, resolution, multi_project_on);
  	object_layer_node.set_camera_param(camera_fx, camera_fy, camera_cx, camera_cy, camera_x, camera_y, camera_z, camera_rx, camera_ry, camera_rz);
  	object_layer_node.set_camera_range_param(image_from_width, image_to_width, image_from_height, image_to_height);

  	while (ros::ok())
		object_layer_node.run();

	return 0;
}

