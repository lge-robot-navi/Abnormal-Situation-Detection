/**
 * @file		image_to_gridmap.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in image_to_gridmap class.
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


#include "osr_map_demos/image_to_gridmap.h"
namespace osr_map {

image_to_gridmap::image_to_gridmap(ros::NodeHandle& _nh)
    : nh_(_nh),
	  period_(10)
{
  read_params();
  read_image();
  pub_map_ = nh_.advertise<grid_map_msgs::GridMap>("/osr_map_background", 1, true);
}

image_to_gridmap::~image_to_gridmap()
{
}

bool image_to_gridmap::read_params()
{
	nh_.param<std::string>("map_frame_id", map_frame_id_, "map");
	nh_.param<float>("length_x", length_x_, 0.03);
	nh_.param<float>("length_y", length_y_, 0.03);
	nh_.param<float>("offset_x", offset_x_, 0.03);
	nh_.param<float>("offset_y", offset_y_, 0.03);
	nh_.param<float>("resolution", resolution_, 0.03);
	nh_.param<std::string>("image_path", path_, "map/map.png");
	return true;
}
bool image_to_gridmap::read_image(){
	cv::Mat image;
	image = cv::imread(path_);
	cv::resize(image, image, cv::Size(length_y_, length_x_));
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	grid_map::Position pos(offset_x_, offset_y_);
	grid_map::GridMapCvConverter::initializeFromImage(image, resolution_, map_, pos);
	map_.setFrameId(map_frame_id_);
	grid_map::GridMapCvConverter::addColorLayerFromImage<unsigned char, 3>(image, "color", map_);
	grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 3>(image, "grey", map_);
}
void image_to_gridmap::run(){
	period_.reset();
	if(pub_map_.getNumSubscribers() > 0)
	{
		grid_map_msgs::GridMap map_msg;
		grid_map::GridMapRosConverter::toMessage(map_, map_msg);
		pub_map_.publish(map_msg);
	}
	period_.sleep();
}

} /* namespace */
