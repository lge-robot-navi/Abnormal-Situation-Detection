/**
 * @file		osr_map_agent_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running osr_map_agent_node
 * 				to integrate multiple map layers as elevation layer, thermal layer, and object layers
 * 				into a single layered map. Moreover, it sends the layered map to server through UDP communication.
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


//ROS
#include <ros/ros.h>
#include <ros/callback_queue.h>
// GridMap
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
// Boost
#include <boost/asio.hpp>

// Self-defined
#include "osr_map_agents/agent.h"
#include "osr_map_comm/parsers.h"
#include "osr_map_comm/packets.h"
#include "osr_map_comm/comm.h"

int main(int argc, char **argv) {
	// Node Generation
	ros::init(argc, argv, "osr_map_agent_node");

	ros::NodeHandle nh("~");

	std::string robot_frame_id, map_frame_id, world_frame_id;
	float length_x, length_y, resolution;
	float period;
	bool elevation_layer_on, thermal_layer_on, object_layer_on;
	std::string elevation_layer_topic, thermal_layer_topic, object_layer_topic;
	std::string pose_topic, map_topic;
	std::string ip_addr;
    int stack_size;
    int robot_id;
	nh.param<int>("robot_id", robot_id, 1);
	nh.param<float>("period", period, 10);
	nh.param<std::string>("robot_frame_id", robot_frame_id, "mobile");
	nh.param<std::string>("map_frame_id", map_frame_id, "osr_map");
	nh.param<std::string>("world_frame_id", world_frame_id, "world");

	nh.param<float>("length_x", length_x, 50);
	nh.param<float>("length_y", length_y, 50);
	nh.param<float>("resolution", resolution, 0.5);

    nh.param<int>("stack_size", stack_size, 1);

	nh.param<bool>("elevation_layer_on", elevation_layer_on, false);
	nh.param<bool>("thermal_layer_on", thermal_layer_on, false);
	nh.param<bool>("object_layer_on", object_layer_on, false);

	nh.param<std::string>("elevation_layer_topic", elevation_layer_topic, "/elevation_layer");
	nh.param<std::string>("thermal_layer_topic", thermal_layer_topic, "/thermal_layer");
	nh.param<std::string>("object_layer_topic", object_layer_topic, "/object_layer");

	nh.param<std::string>("pose_topic", pose_topic, "/robot_odom");
	nh.param<std::string>("map_topic", map_topic, "/osr/agent_map");

	nh.param<std::string>("udp_ip_address", ip_addr, "127.0.0.1");
	int local_id;
	int data_type;
	nh.param<int>("local_id", local_id, 8888);
	nh.param<int>("data_type", data_type, 8888);
	unsigned int port = osr_map::get_port(local_id, data_type, robot_id);

	osr_map::agent agent_node(nh, period, robot_frame_id, map_frame_id);
	agent_node.set_map_param(length_x, length_y, resolution, stack_size);
	std::map<std::string, std::pair<float, float>> min_max_table;
	float min_elevation, max_elevation;
	float min_person_flow_x, max_person_flow_x;
	float min_person_flow_y, max_person_flow_y;
	float min_car_flow_x, max_car_flow_x;
	float min_car_flow_y, max_car_flow_y;
	float min_thermal, max_thermal;

	nh.param<float>("min_elevation", min_elevation, -0.5);
	nh.param<float>("max_elevation", max_elevation, 2.0);
	nh.param<float>("min_person_flow_x", min_person_flow_x, -5.0);
	nh.param<float>("max_person_flow_x", max_person_flow_x, 5.0);
	nh.param<float>("min_person_flow_y", max_person_flow_y, -5.0);
	nh.param<float>("max_person_flow_y", max_person_flow_y, 5.0);
	nh.param<float>("min_car_flow_x", min_car_flow_x, -30.0);
	nh.param<float>("max_car_flow_x", max_car_flow_x, 30.0);
	nh.param<float>("min_car_flow_y", min_car_flow_y, -30.0);
	nh.param<float>("max_car_flow_y", max_car_flow_y, 30.0);
	nh.param<float>("min_thermal", min_thermal, 0.0);
	nh.param<float>("max_thermal", max_thermal, 100.0);

	if(elevation_layer_on){
		agent_node.set_elevation_layer_on(elevation_layer_topic);
		min_max_table.insert({"elevation_update", {0, 0}});
		min_max_table.insert({"elevation", {min_elevation, max_elevation}});
	}
	if(thermal_layer_on){
		agent_node.set_thermal_layer_on(thermal_layer_topic);
		min_max_table.insert({"thermal_update", {0, 0}});
		min_max_table.insert({"thermal", {min_thermal, max_thermal}});
	}
	if(object_layer_on){
		agent_node.set_object_layer_on(object_layer_topic);
		min_max_table.insert({"object_update", {0, 0}});
		min_max_table.insert({"person_number", {0, 0}});
		min_max_table.insert({"person_posture", {0, 0}});
		min_max_table.insert({"person_flow_x", {min_person_flow_x, max_person_flow_x}});
		min_max_table.insert({"person_flow_y", {min_person_flow_y, max_person_flow_y}});
		min_max_table.insert({"car_number", {0, 0}});
		min_max_table.insert({"car_flow_x", {min_car_flow_x, max_car_flow_x}});
		min_max_table.insert({"car_flow_y", {min_car_flow_y, max_car_flow_y}});
	}

	agent_node.set_pose_on(pose_topic);

	boost::asio::io_service io_srv;
	osr_map::udp_sender udp_sender(io_srv, ip_addr, port);
	ros::Publisher pub_map = nh.advertise < grid_map_msgs::GridMap > (map_topic, 5, true);

	ros::Rate rate(period);
	ros::AsyncSpinner spinner(0);
	spinner.start();
	int seq = 0;
	while (ros::ok()){
		agent_node.reset();
		grid_map::GridMap grid_map;
		if(!agent_node.estimate_map(grid_map)){
			rate.sleep();
		    continue;
		}
		try{
			osr_map::light_layered_map layered_map;
			osr_map::convert_light_map_from_grid_map(robot_id, robot_frame_id, seq, grid_map, min_max_table, layered_map);
			size_t byte_transferred = udp_sender.send(layered_map);
			seq++;
			ROS_INFO_STREAM("agent : [ "<< byte_transferred << " ] bytes sent to server.");
			layered_map.clear();
		}
		catch (const std::exception& ex) {
	        ROS_ERROR("agent : %s", ex.what());
		}

		// map to publish
		if (pub_map.getNumSubscribers() > 0 && !grid_map.getLayers().empty()) {
			grid_map_msgs::GridMap map_msg;
			grid_map::GridMapRosConverter::toMessage(grid_map, map_msg);
			pub_map.publish(map_msg);
		}

		rate.sleep();
	}

	spinner.stop();
	return 0;
}
