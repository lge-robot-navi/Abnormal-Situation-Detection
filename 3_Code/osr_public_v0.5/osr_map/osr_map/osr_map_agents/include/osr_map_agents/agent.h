/**
 * @file		agent.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file include agent class to integrate multiple map layers
 * 				like elevation, thermal, and object into a single map
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

#ifndef AGENT_H_
#define AGENT_H_

// ros
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <nav_msgs/Odometry.h>

// eigen
#include <Eigen/Dense>

// grid_map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>


namespace osr_map{
	/**
	 *  @class 		agent
	 *  @brief 		This class  integrate multiple map layers like elevation, thermal, and object into a single map
	 *  @since		Mar 22, 2019
	 */
	class agent{
	public:
		agent(ros::NodeHandle _nh, int _period, std::string _robot_frame_id, std::string _map_frame_id);
		~agent(){};

		/**
		* @fn		set_map_param
		* @brief	set map parameter related to size and resolution
		* @param	_length_x 			x-directional size
		* @param	_length_y			y-directional size
		* @param	_resolution			resolution
		* @param	_stack_size			the number of layers to project onto layered map for a single type of map layer
		*/
		void set_map_param(float _length_x, float _length_y, float _resolution, int _stack_size);

		/**
		* @fn		set_elevation_layer_on
		* @brief	set topic name and member variable to handling elevation_layer
		* @param	_layer_topic	elevation layer topic
		*/
		void set_elevation_layer_on(std::string _layer_topic);

		/**
		* @fn		set_thermal_layer_on
		* @brief	set topic name and member variable to handling thermal_layer
		* @param	_layer_topic	thermal_layer topic
		*/
		void set_thermal_layer_on(std::string _layer_topic);

		/**
		* @fn		set_object_layer_on
		* @brief	set topic name and member variable to handling object_layer
		* @param	_layer_topic	object_layer topic
		*/
		void set_object_layer_on(std::string _layer_topic);


		/**
		* @fn		set_pose_on
		* @brief	set topic name and member variable to handling pose
		* @param	_layer_topic	pose topic
		*/
		void set_pose_on(std::string _pose_topic);


		/**
		* @fn		estimate_map
		* @brief	integrate handling map layers by considering current pose to a single layered map.
		* @param	_map	single layered map
		*/
		bool estimate_map(grid_map::GridMap& _map);


		/**
		* @fn		project_map
		* @brief	project source map to destination map.
		* @param	_src_map	source map
		* @param	_dst_map	destination map
		*/
		bool project_map(grid_map::GridMap& _src_map, grid_map::GridMap& _dst_map);

		/**
		* @fn            reset
		* @brief         reset member variables for measuring periodic time.
		*/
		void reset();

	private:
		/**
		* @fn            subscribe_pose_callback
		* @brief         subscribe pose and save to member variable.
		* @param		 _msg : pose message
		*/
		void subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg);

		/**
		* @fn            subscribe_object_layer_callback
		* @brief         subscribe object layer and save to member variable.
		* @param		 _msg : object_layer message
		*/
		void subscribe_object_layer_callback(const grid_map_msgs::GridMap::ConstPtr& _msg);

		/**
		* @fn            subscribe_elevation_layer_callback
		* @brief         subscribe elevation_layer and save to member variable.
		* @param		 _msg : elevation_layer message
		*/
		void subscribe_elevation_layer_callback(const grid_map_msgs::GridMap::ConstPtr& _msg);

		/**
		* @fn            subscribe_thermal_layer_callback
		* @brief         subscribe thermal_layer and save to member variable.
		* @param		 _msg : thermal_layer message
		*/
		void subscribe_thermal_layer_callback(const grid_map_msgs::GridMap::ConstPtr& _msg);

		ros::NodeHandle nh_;
		std::string robot_frame_id_, map_frame_id_;
		ros::Rate period_;
		ros::Time curr_time_, prev_time_;
		float length_x_, length_y_, resolution_;
		int stack_size_;
		bool is_in_object_layer_, is_in_elevation_layer_, is_in_thermal_layer_, is_in_pose_;
		nav_msgs::Odometry pose_;

		std::vector<grid_map_msgs::GridMap> object_layer_msgs_;
		std::vector<grid_map_msgs::GridMap> thermal_layer_msgs_;
		std::vector<grid_map_msgs::GridMap> elevation_layer_msgs_;
		grid_map_msgs::GridMap object_layer_msg_;
		grid_map_msgs::GridMap thermal_layer_msg_;
		grid_map_msgs::GridMap elevation_layer_msg_;
		bool object_layer_on_, elevation_layer_on_, thermal_layer_on_;
		std::vector<std::string> layers_;
		ros::Subscriber sub_object_layer_, sub_thermal_layer_, sub_elevation_layer_, sub_pose_;
	};
}



#endif /* AGENT_H_ */
