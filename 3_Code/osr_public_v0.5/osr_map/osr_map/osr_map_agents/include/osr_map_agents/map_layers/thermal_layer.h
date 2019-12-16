/**
 * @file		thermal_layer.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes thermal_layer class for building and publishing thermal_layer of mobile agent by subscribing thermal image, 3D point cloud and pose
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

#ifndef THERMAL_LAYER_H_
#define THERMAL_LAYER_H_


// ros
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/CameraInfo.h>
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
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <tf/tf.h>

namespace osr_map{
	/**
	 *  @class 			thermal_layer
	 *  @brief 			This class builds and publish thermal_layer of mobile agent by subscribing thermal image, 3D point cloud and pose
	 *  @since			Mar 22, 2019
	 */
	class thermal_layer{
	public:
		thermal_layer(ros::NodeHandle _nh, int _period, int _robot_id, std::string _robot_frame_id, std::string _map_frame_id);
		~thermal_layer(){};
		/**
		* @fn		set_comm
		* @brief	set topic name for ROS subscriber and publisher
		* @param	_layer_topic 			map layer topic name for publishing
		* @param	_pose_topic				pose topic name for subscribing
		* @param	_thermal_image_topic	thermal image topic name for subscribing
		* @param	_pointcloud_topic		point cloud topic name for subscribing
		*/
		void set_comm(std::string _layer_topic, std::string _pose_topic, std::string _thermal_image_topic, std::string _pointcloud_topic, std::string _thermal_to_lidar_info_topic);

		/**
		* @fn		set_map_param
		* @brief	set map parameter related to size and resolution
		* @param	_length_x 			x-directional size
		* @param	_length_y			y-directional size
		* @param	_resolution			resolution
		*/
		void set_map_param(float _length_x, float _length_y, float _resolution);

		/**
		* @fn		set_thermal_param
		* @brief	set temperature parameter to limit the range of temperature
		* @param	_min_thermal	minimum temperature
		* @param	_max_thermal	maximum temperature
		*/
		void set_thermal_param(float _min_thermal, float _max_thermal);

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


		/**
		* @fn            subscribe_thermal_callback
		* @brief         subscribe thermal image and save to member variable.
		* @param		 _msg : thermal image
		*/
		void subscribe_thermal_callback(const sensor_msgs::Image::ConstPtr& _msg);
		/**
		* @fn            subscribe_thermal_to_lidar_info_callback
		* @brief         subscribe calibration information between thermal camera (image) and 3D LiDAR (point cloud) image and save to member variable.
		* @param		 _msg : thermal image
		*/
		void subscribe_thermal_to_lidar_info_callback(const sensor_msgs::CameraInfo::ConstPtr& _msg);

		ros::NodeHandle nh_;
		int robot_id_;
		std::string robot_frame_id_, map_frame_id_;
		ros::Rate period_;
		ros::Time curr_time_, prev_time_;
		float length_x_, length_y_, resolution_;
		float min_thermal_, max_thermal_;
		float min_range_, max_range_;
		bool is_in_thermal_to_lidar_info_, is_in_pointcloud_, is_in_thermal_, is_in_pose_;
		ros::Subscriber sub_pointcloud_;
		ros::Subscriber sub_thermal_;
		ros::Subscriber sub_thermal_to_lidar_info_;
		ros::Subscriber sub_pose_;
		ros::Publisher pub_layer_;
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_ptr_;
		cv::Mat thermal_image_;
		nav_msgs::Odometry pose_;
		Eigen::MatrixXf thermal_to_lidar_;
	};
}


#endif /* THERMAL_LAYER_H_ */
