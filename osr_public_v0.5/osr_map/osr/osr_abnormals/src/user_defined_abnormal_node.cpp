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
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
#include "osr_abnormals/user_defined_abnormal.h"

int main(int argc, char **argv) {
	ros::init(argc, argv, "user_defined_abnormal_node");
	ros::NodeHandle nh("~");

	float period;
	std::string map_frame_id, world_frame_id;
	std::string map_topic, agent_info_topic, abnormal_topic;
	bool illegal_parking_on, high_thermal_on, high_elevation_on;
	bool viz_on;
	float min_car_speed, max_temperature, max_height;

	nh.param<float>("period", period, 10);
	nh.param < std::string > ("map_frame_id", map_frame_id, "osr_map");
	nh.param < std::string > ("world_frame_id", world_frame_id, "world");
	nh.param < std::string > ("map_topic", map_topic, "/osr_map");
	nh.param < std::string > ("agent_info_topic", agent_info_topic, "/osr_agent_info");
	nh.param < std::string > ("abnormal_topic", abnormal_topic, "/osr_user_defined_abnormals");
	nh.param<bool>("illegal_parking_on", illegal_parking_on, false);
	nh.param<bool>("high_thermal_on", high_thermal_on, false);
	nh.param<bool>("high_elevation_on", high_elevation_on, false);
	nh.param<bool>("viz_on", viz_on, false);
	nh.param<float>("min_car_speed", min_car_speed, 1.0);
	nh.param<float>("max_temperature", max_temperature, 60.0);
	nh.param<float>("max_height", max_height, 3.0);

    std::vector<float> illegal_parking_area;
    std::vector<float> high_thermal_area;
    std::vector<float> high_elevation_area;
    nh.param("illegal_parking_area", illegal_parking_area, std::vector<float>());
    nh.param("high_thermal_area", high_thermal_area, std::vector<float>());
    nh.param("high_elevation_area", high_elevation_area, std::vector<float>());


	ros::Rate rate(period);
	osr::user_defined_abnormal abnormal(nh, period, map_frame_id, world_frame_id);
	abnormal.set_comm(map_topic, agent_info_topic, abnormal_topic);
	if(illegal_parking_on){abnormal.set_illegal_parking_on(min_car_speed, illegal_parking_area);}
	if(high_thermal_on){abnormal.set_high_thermal_on(max_temperature, high_thermal_area);}
	if(high_elevation_on){abnormal.set_high_elevation_on(max_height, high_elevation_area);}
	if(viz_on){abnormal.set_viz_on();}
    while(ros::ok()){
    	rate.reset();
    	abnormal.reset();
    	ros::spinOnce();
    	abnormal.detect_abnormals();
    	abnormal.publish_abnormals();
    	abnormal.visualize();
    	rate.sleep();
    }
	return 0;
}
