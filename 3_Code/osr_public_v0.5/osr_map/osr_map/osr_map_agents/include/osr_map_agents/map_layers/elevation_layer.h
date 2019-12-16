/**
 * @file		elevation_layer.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes elevation_layer class for building and publishing elevation_layer of mobile agent by subscribing 3D point cloud and pose
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

#ifndef ELEVATION_LAYER_H_
#define ELEVATION_LAYER_H_


// ros
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>

// eigen
#include <Eigen/Dense>

// grid_map
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

// pcl
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

namespace osr_map{
	/**
	 *  @class 			elevation_layer
	 *  @brief 			This class builds and publish elevation_layer of mobile agent by subscribing 3D point cloud and pose
	 *  @since			Mar 22, 2019
	 */
	class elevation_layer{
	public:
		elevation_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id);
		~elevation_layer(){};
		/**
		* @fn		set_comm
		* @brief	set topic name for ROS subscriber and publisher
		* @param	_layer_topic 		map layer topic name for publishing
		* @param	_pose_topic			pose topic name for subscribing
		* @param	_pointcloud_topic	3D point cloud topic name for subscribing
		*/
		void set_comm(std::string _layer_topic, std::string _pose_topic, std::string _pointcloud_topic);

		/**
		* @fn		set_map_param
		* @brief	set map parameter related to size and resolution
		* @param	_length_x 		x-directional size
		* @param	_length_y		y-directional size
		* @param	_resolution		resolution
		*/
		void set_map_param(float _length_x, float _length_y, float _resolution);

		/**
		* @fn		set_z_param
		* @brief	set elevation-related parameters
		* @param	_offset_z 		z-directional offset
		* @param	_min_z			minimum threshold of z
		* @param	_max_z			maximum threshold of z
		* @param	_thickness		the ground thickness
		*/
		void set_z_param(float _offset_z, float _min_z, float _max_z, float _thickness);

		/**
		* @fn		set_range_param
		* @brief	set range-related parameters to filter out point cloud
		* @param	_min_range 		minimum threshold of range
		* @param	_max_range		maximum threshold of range
		*/
		void set_range_param(float _min_range, float _max_range);

		/**
		* @fn            reset
		* @brief         reset member variables for measuring periodic time.
		*/
		void reset();


		/**
		* @fn            run
		* @brief         periodically build elevation layers by considering the parameters and publish those.
		*/
		void run();

	private:
		/**
		* @fn            subscribe_pose_callback
		* @brief         subscribe pose and save to member variable.
		* @param		 _msg : pose message
		*/
		void subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg);

		/**
		* @fn            subscribe_pointcloud_callback
		* @brief         subscribe 3D point cloud and save to member variable.
		* @param		 _msg : 3D point cloud
		*/
		void subscribe_pointcloud_callback(const sensor_msgs::PointCloud2::ConstPtr& _msg);

		ros::NodeHandle nh_;
		int robot_id_;
		std::string robot_frame_id_, map_frame_id_;
		ros::Rate period_;
		ros::Time curr_time_, prev_time_;
		float length_x_, length_y_, resolution_;
		float offset_z_, min_z_, max_z_, thickness_;
		float min_range_, max_range_;
		bool is_in_pointcloud_, is_in_pose_;
		ros::Subscriber sub_pointcloud_;
		ros::Subscriber sub_pose_;
		ros::Publisher pub_layer_;
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_ptr_;
		nav_msgs::Odometry pose_;
	};
}



#endif /* ELEVATION_LAYER_H_ */
