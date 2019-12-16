/**
 * @file		fixed_thermal_layer_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na  (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running fixed_thermal_layer_node
 * 				to build thermal_layer of fixed agent
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

#include "osr_map_agents/map_layers/fixed_thermal_layer.h"

int main(int argc, char **argv)
{
  	// Node Generation
  	ros::init(argc, argv, "fixed_thermal_layer");

  	ros::NodeHandle nh("~");

  	std::string robot_frame_id, map_frame_id, world_frame_id;
  	std::string layer_topic, pose_topic, depth_image_topic;
  	float length_x, length_y, resolution;
  	float offset_z, max_z, thickness;
  	float period;

  	int robot_id;
  	nh.param<int>("robot_id", robot_id, 1);

  	nh.param<float>("period", period, 10);
  	nh.param<float>("length_x", length_x, 50);
  	nh.param<float>("length_y", length_y, 50);
  	nh.param<float>("resolution", resolution, 0.5);

  	nh.param<float>("offset_z", offset_z, 1.0);
  	nh.param<float>("max_z", max_z, 5.0);
  	nh.param<float>("thickness", thickness, 0.3);

  	float min_dist, max_dist;
  	nh.param<float>("min_dist", min_dist, 1.0);
  	nh.param<float>("max_dist", max_dist, 5.0);


    // Read parameters for 'RGBDMapper'
  	float camera_fx, camera_fy, camera_cx, camera_cy, camera_x, camera_y, camera_z, camera_rx, camera_ry, camera_rz;
  	nh.param<float>("camera_fx",     camera_fx, 612.4);
  	nh.param<float>("camera_fy",     camera_fy, 612.4);
  	nh.param<float>("camera_cx",     camera_cx, 320.0);
  	nh.param<float>("camera_cy",     camera_cy, 240.0);
  	nh.param<float>("camera_x", camera_x, 3.200);
  	nh.param<float>("camera_y", camera_y, 3.580);
  	nh.param<float>("camera_z", camera_z, 2.390);
  	float deg_rx, deg_ry, deg_rz;
	nh.param<float>("camera_tilt",   deg_rx, -23.6);
	nh.param<float>("camera_pan",    deg_ry, -140.0);
	nh.param<float>("camera_roll",   deg_rz, -1.0);
	camera_rx = cx::cvtDeg2Rad(deg_rx);
	camera_ry = cx::cvtDeg2Rad(deg_ry);
	camera_rz = cx::cvtDeg2Rad(deg_rz);


  	nh.param<std::string>("layer_topic", layer_topic, "/thermal_layer");
  	nh.param<std::string>("pose_topic", pose_topic, "/amcl_pose");
  	nh.param<std::string>("depth_image_topic", depth_image_topic, "/velodyne_points");

  	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
  	nh.param<std::string>("map_frame_id", map_frame_id, "osr_map");
  	nh.param<std::string>("world_frame_id", world_frame_id, "world");

  	osr_map::fixed_thermal_layer thermal_layer_node(nh, period, robot_id, robot_frame_id, map_frame_id);
  	thermal_layer_node.set_comm(layer_topic, pose_topic, depth_image_topic);
  	thermal_layer_node.set_map_param(length_x, length_y, resolution);
  	thermal_layer_node.set_z_param(offset_z, max_z, thickness);
  	thermal_layer_node.set_distance_param(min_dist, max_dist);
  	thermal_layer_node.set_camera_param(camera_fx, camera_fy, camera_cx, camera_cy, camera_x, camera_y, camera_z, camera_rx, camera_ry, camera_rz);

  	while (ros::ok())
		thermal_layer_node.run();

	return 0;
}
