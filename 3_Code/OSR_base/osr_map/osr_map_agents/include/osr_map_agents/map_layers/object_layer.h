/**
 * @file		object_layer.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes object_layer class for building and publishing object_layers of mobile agent by subscribing tracks like pedestrian and car, and pose
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

#ifndef OBJECT_LAYER_H_
#define OBJECT_LAYER_H_


// ros
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>

// eigen
#include <Eigen/Dense>

// grid_map
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

// osr_map
#include <osr_msgs/Track.h>
#include <osr_msgs/Tracks.h>

namespace osr_map{
	/**
	 *  @class 			object_layer
	 *  @brief 			This class builds and publish object_layers of mobile agent by subscribing tracks like pedestrian and car, and pose
	 *  @since			Mar 22, 2019
	 */
	class object_layer{
	public:
		object_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id);
		~object_layer(){};

		/**
		* @fn		set_comm
		* @brief	set topic name for ROS subscriber and publisher
		* @param	_layer_topic 	map layer topic name for publishing
		* @param	_object_topic	tracks topic name for subscribing
		*/
		void set_comm(std::string _layer_topic, std::string _object_topic);

		/**
		* @fn		set_map_param
		* @brief	set map parameter related to size and resolution
		* @param	_length_x 			x-directional size
		* @param	_length_y			y-directional size
		* @param	_resolution			resolution
		* @param	_multi_project_on	selection whether multiple objects can be projected on a single grid cell.
		*/
		void set_map_param(float _length_x, float _length_y, float _resolution, bool _multi_project_on);


		/**
		* @fn		set_update_param
		* @brief	set update parameter to build update_layer representing updated cells within parameters.
		* @param	_min_theta 		minimum theta
		* @param	_max_theta		maximum theta
		* @param	_min_range		minimum range
		* @param	_max_range		maximum range
		*/
		void set_update_param(float _min_theta, float _max_theta, float _min_range, float _max_range);


		/**
		* @fn            reset
		* @brief         reset member variables for measuring periodic time.
		*/
		void reset();

		/**
		* @fn            run
		* @brief         periodically build object layers by considering the parameters and publish those.
		*/
		void run();

	private:
		/**
		* @fn            subscribe_object_callback
		* @brief         subscribe tracks and save to member variable.
		* @param		 _msg : tracks message
		*/
		void subscribe_object_callback(const osr_msgs::Tracks::ConstPtr& _msg);

		ros::NodeHandle nh_;
		int robot_id_;
		std::string robot_frame_id_, map_frame_id_;
		ros::Rate period_;
		ros::Time curr_time_, prev_time_;
		float length_x_, length_y_, resolution_;
		float min_theta_, max_theta_, min_range_, max_range_;
		bool multi_project_on_;
		osr_msgs::Tracks objects_;
		nav_msgs::Odometry odom_;
		bool is_in_tracks_;
		ros::Subscriber sub_object_;
		ros::Publisher pub_layer_;
	};
}




#endif /* OBJECT_LAYER_H_ */
