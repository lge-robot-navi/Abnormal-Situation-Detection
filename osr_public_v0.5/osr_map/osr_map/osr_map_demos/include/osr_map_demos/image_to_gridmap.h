/**
 * @file		image_to_grid_map.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes image_to_grid_map class for visualizing RGB image as grid map through occupancy grid map and point cloud on RVIZ
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

#ifndef OSR_MAP_OSR_MAP_OSR_MAP_DEMOS_INCLUDE_IMAGE_TO_GRIDMAP_H_
#define OSR_MAP_OSR_MAP_OSR_MAP_DEMOS_INCLUDE_IMAGE_TO_GRIDMAP_H_

#pragma once

// ROS
#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace osr_map {
/**
 *  @class 			image_to_gridmap
 *  @brief 			This class visualize RGB image as grid map through occupancy grid map and point cloud on RVIZ
 *  @since			Mar 22, 2019
 */
class image_to_gridmap
{
 public:

	image_to_gridmap(ros::NodeHandle& _nh);
	virtual ~image_to_gridmap();
	/**
	* @fn		read_params
	* @brief	load ROS parameters
	* @return 	success or fail
	*/
	bool read_params();

	/**
	* @fn		read_image
	* @brief	load image and convert to grid map
	* @return 	success or fail
	*/
	bool read_image();

	/**
	* @fn		run
	* @brief	publish grid map
	*/
	void run();
private:
	ros::NodeHandle& nh_;
	ros::Rate period_;
	ros::Publisher pub_map_;
	grid_map::GridMap map_;

	float length_x_;
	float length_y_;
	float offset_x_;
	float offset_y_;
	float resolution_;

	std::string map_frame_id_;
	std::string path_;
};

} /* namespace */


#endif /* OSR_MAP_OSR_MAP_OSR_MAP_DEMOS_INCLUDE_IMAGE_TO_GRIDMAP_H_ */
