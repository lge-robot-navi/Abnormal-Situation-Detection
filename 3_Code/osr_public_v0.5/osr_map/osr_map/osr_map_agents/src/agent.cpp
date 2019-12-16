/**
 * @file		agent.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in agent class.
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
#include "osr_map_agents/agent.h"
osr_map::agent::agent(ros::NodeHandle _nh, int _period, std::string _robot_frame_id, std::string _map_frame_id) :
		nh_(_nh), period_(_period), robot_frame_id_(_robot_frame_id), map_frame_id_(_map_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
	object_layer_on_ = false;
	elevation_layer_on_ = false;
	thermal_layer_on_ = false;
	is_in_object_layer_ = is_in_elevation_layer_ = is_in_thermal_layer_ = is_in_pose_ = false;
}
void osr_map::agent::set_elevation_layer_on(std::string _layer_topic)
{
	sub_elevation_layer_ = nh_.subscribe(_layer_topic, 1, &osr_map::agent::subscribe_elevation_layer_callback, this);
	layers_.push_back("elevation");
	elevation_layer_on_ = true;
}
void osr_map::agent::set_thermal_layer_on(std::string _layer_topic)
{
	sub_thermal_layer_ = nh_.subscribe(_layer_topic, 1, &osr_map::agent::subscribe_thermal_layer_callback, this);
	layers_.push_back("thermal");
	thermal_layer_on_ = true;
}
void osr_map::agent::set_object_layer_on(std::string _layer_topic)
{
	sub_object_layer_ = nh_.subscribe(_layer_topic, 1, &osr_map::agent::subscribe_object_layer_callback, this);

	layers_.push_back("person_posture");
	layers_.push_back("person_number");
	layers_.push_back("person_flow_x");
	layers_.push_back("person_flow_y");
	layers_.push_back("car_number");
	layers_.push_back("car_flow_x");
	layers_.push_back("car_flow_y");
	layers_.push_back("object_update");
	object_layer_on_ = true;
}
void osr_map::agent::set_pose_on(std::string _pose_topic)
{
	sub_pose_ = nh_.subscribe(_pose_topic, 1, &osr_map::agent::subscribe_pose_callback, this);
}
void osr_map::agent::set_map_param(float _length_x, float _length_y, float _resolution, int _stack_size)
{
	length_x_ = _length_x;
	length_y_ = _length_y;
	resolution_ = _resolution;
	stack_size_ = _stack_size;
}
void osr_map::agent::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
	prev_time_ = curr_time_;
}
bool osr_map::agent::project_map(grid_map::GridMap& _src_map, grid_map::GridMap& _dst_map)
{
	std::vector<std::string> dst_layers = _dst_map.getLayers();
	std::vector<std::string> src_layers = _src_map.getLayers();

	for (grid_map::GridMapIterator it(_src_map); !it.isPastEnd(); ++it)
	{
		grid_map::Position pos;
		_src_map.getPosition(*it, pos);
		if(!_dst_map.isInside(pos)){continue;} // fixed size of map
		grid_map::Index idx;
		_dst_map.getIndex(pos, idx);
		for (const auto& layer : src_layers){
			if(!_src_map.isValid(*it, layer)){continue;}
			_dst_map.at(layer, idx) = _src_map.at(layer, *it);
		}
	}
	return true;
}
bool osr_map::agent::estimate_map(grid_map::GridMap& _map)
{
	_map.setTimestamp(curr_time_.toNSec());
	_map.setGeometry(grid_map::Length(length_x_, length_y_), resolution_, grid_map::Position(pose_.pose.pose.position.x, pose_.pose.pose.position.y));
	_map.setFrameId(map_frame_id_);
	if(!is_in_pose_)
	{
		ROS_ERROR("agent : NO TOPIC INPUT FOR POSE.");
		return false;
	}
	if (object_layer_on_ && is_in_object_layer_) {
		grid_map::GridMap object_layer;
		grid_map::GridMapRosConverter::fromMessage(object_layer_msg_, object_layer);
		if(!_map.addDataFrom(object_layer, false, true, true)){
			ROS_ERROR("agent: Failed to copy object layer");
		}
	}
	if (elevation_layer_on_ && is_in_elevation_layer_) {
		grid_map::GridMap elevation_layer;
		grid_map::GridMapRosConverter::fromMessage(elevation_layer_msg_, elevation_layer);
		if(!_map.addDataFrom(elevation_layer, false, true, true)){
			ROS_ERROR("agent: Failed to copy elevation layer");
		}
	}
	if (thermal_layer_on_ && is_in_thermal_layer_) {
		grid_map::GridMap thermal_layer;
		grid_map::GridMapRosConverter::fromMessage(thermal_layer_msg_, thermal_layer);
		if(!_map.addDataFrom(thermal_layer, false, true, true)){
			ROS_ERROR("agent: Failed to copy thermal layer");
		}
	}
	if(_map.getLayers().empty()){
		return false;
	}
	return true;
}
void osr_map::agent::subscribe_object_layer_callback(const grid_map_msgs::GridMap::ConstPtr& _msg) {
	is_in_object_layer_ = true;
	object_layer_msg_ = *_msg;
}
void osr_map::agent::subscribe_elevation_layer_callback(const grid_map_msgs::GridMap::ConstPtr& _msg) {
	is_in_elevation_layer_ = true;
	elevation_layer_msg_ = *_msg;
}
void osr_map::agent::subscribe_thermal_layer_callback(const grid_map_msgs::GridMap::ConstPtr& _msg) {
	is_in_thermal_layer_ = true;
	thermal_layer_msg_ = *_msg;
}
void osr_map::agent::subscribe_pose_callback(const nav_msgs::Odometry::ConstPtr& _msg) {
	is_in_pose_ = true;
	pose_ = *_msg;
}
