/**
 * @file		short_term_server_node.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes main function for running short_term_server_node
 * 				to receive local layered maps through UDP communication from fixed and mobile agents and
 * 				to integrate the local layered maps to the global layered map
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
#include <visualization_msgs/MarkerArray.h>
#include "osr_map_comm/comm.h"
#include "osr_map_comm/packets.h"
#include "osr_map_comm/parsers.h"
#include "osr_map_server/map_manager.h"
#include "osr_msgs/AgentInfos.h"
#include "osr_msgs/AgentInfo.h"
#include "osr_msgs/MapInfo.h"

int main(int argc, char **argv) {
	ros::init(argc, argv, "short_term_server_node");
	ros::NodeHandle nh("~");

	float period;
	float position_x, position_y, length_x, length_y, resolution;
	std::string world_frame_id, map_frame_id;
	std::string map_path, map_name, map_topic;
	bool object_map_on, elevation_map_on, thermal_map_on;
	bool load_map_on, extend_map;
	bool viz_on;

	nh.param<float>("period", period, 10);
	nh.param < std::string > ("map_frame_id", map_frame_id, "osr_map");
	nh.param < std::string > ("world_frame_id", world_frame_id, "world");

	nh.param<float>("position_x", position_x, 0.0);
	nh.param<float>("position_y", position_y, 0.0);
	nh.param<float>("length_x", length_x, 50.0);
	nh.param<float>("length_y", length_y, 50.0);
	nh.param<float>("resolution", resolution, 0.5);

	nh.param < bool > ("object_map_on", object_map_on, false);
	nh.param < bool > ("elevation_map_on", elevation_map_on, false);
	nh.param < bool > ("thermal_map_on", thermal_map_on, false);

	nh.param < bool > ("load_map_on", load_map_on, false);
	nh.param < bool > ("extend_map", extend_map, true);

	nh.param < bool > ("viz_on", viz_on, false);

	int port;
	nh.param<int>("udp_port", port, 12345);


	nh.param < std::string > ("map_path", map_path, "none");
	nh.param < std::string > ("map_name", map_name, "map");
	nh.param < std::string > ("map_topic", map_topic, "/osr_map");

	std::string agent_info_topic, map_info_topic;
	nh.param < std::string > ("agent_info_topic", agent_info_topic, "/osr_agent_info");
	nh.param < std::string > ("map_info_topic", map_info_topic, "/osr_map_info");

	std::vector<std::string> layers;
	std::vector<std::string> update_layers;

	if(object_map_on){
	    layers.push_back("person_posture");
		layers.push_back("person_number");
		layers.push_back("person_flow_x");
		layers.push_back("person_flow_y");
		layers.push_back("car_number");
		layers.push_back("car_flow_x");
		layers.push_back("car_flow_y");
		layers.push_back("object_update");
		update_layers.push_back("object_update");
	}
	if(elevation_map_on){
		layers.push_back("elevation");
		layers.push_back("elevation_update");
		update_layers.push_back("elevation_update");
	}
	if(thermal_map_on){
		layers.push_back("thermal");
		layers.push_back("thermal_update");
		update_layers.push_back("thermal_update");
	}


	boost::asio::io_service io_srv;
	osr_map::udp_receiver receiver(io_srv, static_cast<unsigned int>(port));
	boost::thread receive_thread(boost::bind(&osr_map::udp_receiver::receive_loop, &receiver));

	ros::Time prev_time, curr_time;
	prev_time = curr_time = ros::Time::now();
	ros::Publisher pub_map, pub_viz, pub_agent_info, pub_map_info;
	ros::Rate rate(period);
	pub_map = nh.advertise< grid_map_msgs::GridMap >(map_topic, 1, true);
	pub_agent_info = nh.advertise< osr_msgs::AgentInfos >(agent_info_topic, 1, true);
	pub_map_info = nh.advertise< osr_msgs::MapInfo >(map_info_topic, 1, true);
	pub_viz = nh.advertise< visualization_msgs::MarkerArray >("/osr_map_server_viz", 1, true);

    osr_map::map_manager manager(map_frame_id, world_frame_id);
    if(load_map_on)
    {
    	if(!manager.load_map_bag(map_path, map_name)){
    		manager.initialize_map(grid_map::Position(position_x, position_y), grid_map::Length(length_x, length_y), resolution, layers);
    	}
    }
    else{
    	manager.initialize_map(grid_map::Position(position_x, position_y), grid_map::Length(length_x, length_y), resolution, layers);
    }

	// fixed size and fixed origin
    float running_sec = 0;
    float running_min = 0;
    float running_hour = 0;
    while(ros::ok()){
    	rate.reset();

    	curr_time = ros::Time::now();
    	float time_diff = (float)((curr_time - prev_time).toNSec() * (1e-6));
    	prev_time = curr_time;
    	running_sec += time_diff/1000;
    	if(running_sec > 60){running_min++;running_sec -= 60;}
    	if(running_min > 60){running_hour++;running_min -= 60;}

        std::vector<int> robot_ids;
        std::map<int, std::string> robot_names;
        std::map<int, int> robot_seqs;
        std::map<int, grid_map::Position> robot_poses;
        std::map<int, grid_map::GridMap> robot_maps;


		std::vector<osr_map::light_layered_map> layered_maps;
		if(!receiver.get_all_messages(layered_maps)){
	    	rate.sleep();
			continue;
		}
    	ROS_INFO_STREAM("osr_map_server : loop time : " << time_diff << "m, running time : " << running_hour << "H, " << running_min << "M, " << static_cast<int>(running_sec) << "S.");
		manager.reset_update_layers(update_layers);
		for(std::vector<osr_map::light_layered_map>::reverse_iterator rit = layered_maps.rbegin(); rit != layered_maps.rend(); ++rit){
			std::string agent_id;
			int seq;
			int robot_id;
    	    grid_map::GridMap grid_map;
			osr_map::convert_light_map_to_grid_map(*rit, robot_id, agent_id, seq, grid_map);

			std::vector<int>::iterator it_robot = std::find(robot_ids.begin(), robot_ids.end(), robot_id);
			if (it_robot != robot_ids.end()){
				if(robot_seqs[robot_id] > seq){continue;}
				robot_seqs[robot_id] = seq;
				robot_names[robot_id] = agent_id;
				robot_maps[robot_id] = grid_map;
				robot_poses[robot_id] = grid_map.getPosition();
			}
			else{
				robot_ids.push_back(robot_id);
				robot_seqs.insert({robot_id, seq});
				robot_names.insert({robot_id, agent_id});
				robot_maps.insert({robot_id, grid_map});
				robot_poses.insert({robot_id, grid_map.getPosition()});
			}
		}

		for(std::vector<int>::iterator it = robot_ids.begin(); it != robot_ids.end(); ++it){
			if(manager.overwrite_map(robot_maps[*it])){
					ROS_INFO_STREAM("[ " << robot_names[*it] << " ]'s [ " << robot_seqs[*it] << " ] th map at [ " << robot_maps[*it].getTimestamp() << " ] is received.");
			}
		}
		// remove light_layered_map vector
		for(std::vector<osr_map::light_layered_map>::iterator it = layered_maps.begin(); it != layered_maps.end(); ++it){it->clear();}
		layered_maps.erase(layered_maps.begin(), layered_maps.end());
    	grid_map::GridMap map = manager.get_map();

    	// map to publish
    	if (pub_agent_info.getNumSubscribers() > 0) {
    		osr_msgs::AgentInfos agent_infos_msg;
    		agent_infos_msg.header.frame_id;
    		agent_infos_msg.header.stamp;
    		agent_infos_msg.header.seq;

    		for(std::vector<int>::iterator it = robot_ids.begin(); it != robot_ids.end(); ++it){
    			osr_msgs::AgentInfo agent_info_msg;
    			agent_info_msg.id = *it;
				agent_info_msg.type = 1;
				agent_info_msg.timestamp = robot_maps[*it].getTimestamp();
    			agent_info_msg.pos_x = robot_poses[*it](0);
    			agent_info_msg.pos_y = robot_poses[*it](1);
    			agent_infos_msg.agent_infos.push_back(agent_info_msg);
    		}
    		pub_agent_info.publish(agent_infos_msg);
    	}
    	// map to publish
    	if (pub_map_info.getNumSubscribers() > 0) {
    		osr_msgs::MapInfo map_info_msg;
    		map_info_msg.header.frame_id = map.getFrameId();
    		map_info_msg.header.stamp.fromNSec(map.getTimestamp());

    		map_info_msg.resolution = map.getResolution();
    		map_info_msg.length_x = map.getLength()(0);
    		map_info_msg.length_y = map.getLength()(1);
    		pub_map_info.publish(map_info_msg);
    	}
    	// map to publish
    	if (pub_map.getNumSubscribers() > 0) {
    		map.setFrameId(map_frame_id);
    		map.setTimestamp(curr_time.toNSec());
    		grid_map_msgs::GridMap map_msg;
    		grid_map::GridMapRosConverter::toMessage(map, map_msg);
    		pub_map.publish(map_msg);
    	}

    	if(viz_on)
    	{
    		// visualize robot poses
			visualization_msgs::MarkerArray mkrarr;
			int idx = 0;
			for(std::map<int, grid_map::Position>::iterator it = robot_poses.begin(); it != robot_poses.end(); ++it)
			{
				visualization_msgs::Marker robot_marker;
				robot_marker.header.frame_id = map_frame_id.c_str();
				robot_marker.header.stamp = ros::Time::now();
				robot_marker.ns = "pose";
				robot_marker.id = it->first;
				robot_marker.type = visualization_msgs::Marker::CUBE;
				robot_marker.action = visualization_msgs::Marker::MODIFY;

				robot_marker.pose.position.x = (double) it->second(0);
				robot_marker.pose.position.y = (double) it->second(1);
				robot_marker.pose.position.z = 0;
				robot_marker.pose.orientation.x = 0.0;
				robot_marker.pose.orientation.y = 0.0;
				robot_marker.pose.orientation.z = 0.0;
				robot_marker.scale.x = 1.0;
				robot_marker.scale.y = 1.0;
				robot_marker.scale.z = 1.0;
				robot_marker.color.a = 1.0; // Don't forget to set the alpha!
				robot_marker.color.r = 0.0;
				robot_marker.color.g = 1.0;
				robot_marker.color.b = 0.0;
				robot_marker.lifetime = ros::Duration(2.0);

				mkrarr.markers.push_back(robot_marker);

				visualization_msgs::Marker text_marker;
				text_marker.header.frame_id = map_frame_id.c_str();
				text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
				text_marker.action = visualization_msgs::Marker::MODIFY;
				text_marker.ns = "text";
				text_marker.header.stamp = ros::Time::now();
				text_marker.id = idx;

				text_marker.pose.position.x = (double) it->second(0) + 2.0;
				text_marker.pose.position.y = (double) it->second(1);
				text_marker.pose.position.z = (double) 2;
				text_marker.pose.orientation.x = 0.0;
				text_marker.pose.orientation.y = 0.0;
				text_marker.pose.orientation.z = 0.0;
				text_marker.pose.orientation.w = 1.0;

				text_marker.color.r = 0.0;
				text_marker.color.g = 0.0;
				text_marker.color.b = 0.0;
				text_marker.color.a = 1.0;
				text_marker.scale.z = 1;
				text_marker.frame_locked = true;
				text_marker.lifetime = ros::Duration(2.0);
				text_marker.text = it->first;
				mkrarr.markers.push_back(text_marker);
				idx++;
			}
			pub_viz.publish(mkrarr);
    	}

    	rate.sleep();
    }

    if(load_map_on){
    	manager.save_map_bag(map_path, map_name);
    }

	return 0;
}
