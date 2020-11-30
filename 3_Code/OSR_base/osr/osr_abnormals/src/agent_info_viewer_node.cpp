/**
 * @file		user_defined_abnormal_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running user_defined_abnormal_node
 * 				to detect abnormals like illegal parking, high temperature, and high elevation.
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
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include "osr_abnormals/agent_info_viewer.h"

int main(int argc, char **argv) {
	ros::init(argc, argv, "agent_info_viewer_node");
	ros::NodeHandle nh("~");

	float period;
	std::string map_frame_id, world_frame_id;
	std::string map_topic, agent_info_topic, agent_info_viz_topic;
	bool illegal_parking_on, high_thermal_on, high_elevation_on;
	bool viz_on;
	float min_car_speed, max_temperature, max_height;

	nh.param<float>("period", period, 10);
	nh.param < std::string > ("map_frame_id", map_frame_id, "map");
	nh.param < std::string > ("world_frame_id", world_frame_id, "world");
	nh.param < std::string > ("agent_info_topic", agent_info_topic, "/osr_agent_info");
	nh.param < std::string > ("agent_info_viz_topic", agent_info_viz_topic, "/osr_agent_info_viz");



	ros::Rate rate(period);
	osr::agent_info_viewer agent_info_viz(nh, period, map_frame_id, world_frame_id);
	agent_info_viz.set_comm(map_topic, agent_info_topic, agent_info_viz_topic);
    while(ros::ok()){
    	rate.reset();
    	agent_info_viz.reset();
    	ros::spinOnce();
    	agent_info_viz.visualize();
    	rate.sleep();
    }
	return 0;
}
