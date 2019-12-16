/**
 * @file		fixed_thermal_layer.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes fixed_thermal_layer class for building and publishing thermal_layer of fixed agent by subscribing thermal image, 3D point cloud and pose
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

#ifndef FIXED_THERMAL_LAYER_H_
#define FIXED_THERMAL_LAYER_H_


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
#include "osr_map_agents/opencx.hpp"
namespace osr_map{

	/**
	 *  @class 			fixed_thermal_layer
	 *  @brief 			This class builds and publish thermal_layer of fixed agent by subscribing thermal image, 3D point cloud and pose
	 *  @since			Mar 22, 2019
	 */
	class fixed_thermal_layer{
	public:
		fixed_thermal_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id);
		~fixed_thermal_layer(){};

		/**
		* @fn		set_comm
		* @brief	set topic name for ROS subscriber and publisher
		* @param	_layer_topic 	map layer topic name for publishing
		* @param	_pose_topic			pose topic name for subscribing
		* @param	_depth_image_topic	depth image topic name for subscribing
		*/
		void set_comm(std::string _layer_topic, std::string _pose_topic, std::string _depth_image_topic);

		/**
		* @fn		set_map_param
		* @brief	set map parameter related to size and resolution
		* @param	_length_x 			x-directional size
		* @param	_length_y			y-directional size
		* @param	_resolution			resolution
		*/
		void set_map_param(float _length_x, float _length_y, float _resolution);

		/**
		* @fn		set_z_param
		* @brief	set elevation-related parameters
		* @param	_offset_z 		z-directional offset
		* @param	_max_z			maximum threshold of z
		* @param	_thickness		the ground thickness
		*/
		void set_z_param(float _offset_z, float _max_z, float _thickness);

		/**
		* @fn		set_distance_param
		* @brief	set distance-related parameters to filter out point cloud
		* @param	_min_dist 		minimum threshold of distance
		* @param	_max_dist		maximum threshold of distance
		*/
		void set_distance_param(float _min_dist, float _max_dist);

		/**
		* @fn		set_camera_param
		* @brief	set extrinsic and intrinsic camera matrix parameter
		* @param	_fx 		fx of intrinsic camera matrix
		* @param	_fy			fy of intrinsic camera matrix
		* @param	_cx 		cx of intrinsic camera matrix
		* @param	_cy			cy of intrinsic camera matrix
		* @param	_camera_x	x-directional transition of extrinsic camera matrix
		* @param	_camera_y	y-directional transition of extrinsic camera matrix
		* @param	_camera_z	z-directional transition of extrinsic camera matrix
		* @param	_rx 		x-directional transition of extrinsic camera matrix
		* @param	_ry			y-directional transition of extrinsic camera matrix
		* @param	_rz			z-directional transition of extrinsic camera matrix
		*/
		void set_camera_param(float _fx, float _fy, float _cx, float _cy, float _camera_x, float _camera_y, float _camera_z, float _rx, float _ry, float _rz);

		/**
		* @fn            reset
		* @brief         reset member variables for measuring periodic time.
		*/
		void reset();

		/**
		* @fn            run
		* @brief         periodically build thermal layers by considering the parameters and publish those.
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
		* @fn            subscribe_depth_image_callback
		* @brief         subscribe depth image to convert to 3D point cloud and save to member variable.
		* @param		 _msg : depth image
		*/
		void subscribe_depth_image_callback(const sensor_msgs::Image::ConstPtr& _msg);

		ros::NodeHandle nh_;
		int robot_id_;
		std::string robot_frame_id_, map_frame_id_;
		ros::Rate period_;
		ros::Time curr_time_, prev_time_;
		float length_x_, length_y_, resolution_;
		float offset_z_, max_z_, thickness_;
		float min_dist_, max_dist_;
		cv::Matx33d K_inv_, R_;
		ros::Subscriber sub_depth_image_;
		ros::Subscriber sub_pose_;
		ros::Publisher pub_layer_;
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_ptr_;
		cv::Mat depth_image_;
		nav_msgs::Odometry pose_;


//		float camera_fx_, camera_fy_, camera_cx_, camera_cy_, camera_rx_, camera_ry_, camera_rz_;
		float camera_x_, camera_y_, camera_z_;

	};
}



#endif /* FIXED_THERMAL_LAYER_H_ */
