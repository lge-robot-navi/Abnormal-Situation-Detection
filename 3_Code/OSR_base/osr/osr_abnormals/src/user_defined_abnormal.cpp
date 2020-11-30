/**
 * @file		user_defined_abnormal.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in user_defined_abnormal class.
 * @remark
 * @warning
 *
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
#include "osr_abnormals/user_defined_abnormal.h"

osr::user_defined_abnormal::user_defined_abnormal(ros::NodeHandle _nh, int _period, std::string _map_frame_id, std::string _world_frame_id) :
nh_(_nh), period_(_period), map_frame_id_(_map_frame_id), world_frame_id_(_world_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
	is_in_map_ = false;
	is_in_agent_info_ = false;
	illegal_parking_on_ = high_thermal_on_=  high_elevation_on_ = false;
}

void osr::user_defined_abnormal::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
//	ROS_INFO_STREAM("user_defined_abnormal : time_diff : " << time_diff << " ms");
	prev_time_ = curr_time_;
	abnormals_.clear();
}
void osr::user_defined_abnormal::set_comm(std::string _map_topic, std::string _agent_info_topic, std::string _abnormal_topic)
{
	sub_map_ = nh_.subscribe(_map_topic, 1, &osr::user_defined_abnormal::subscribe_map_callback, this);
	sub_agent_info_ = nh_.subscribe(_agent_info_topic, 1, &osr::user_defined_abnormal::subscribe_agent_info_callback, this);
	pub_abnormal_ = nh_.advertise < osr_msgs::Abnormals> (_abnormal_topic, 1);
}

void osr::user_defined_abnormal::set_viz_on(){
	pub_viz_ = nh_.advertise < visualization_msgs::MarkerArray> ("/user_defined_abnormals_viz", 1);
}
void osr::user_defined_abnormal::set_illegal_parking_on(float _min_car_speed, std::vector<float> _area){
	min_car_speed_ = _min_car_speed;
	illegal_parking_on_ = true;
	if(_area.size() < 4){
		ROS_ERROR("user_defined_abnormal : NOT ENOUGH AREA POINTS FOR ILLEGAL PARKING.");
		illegal_parking_on_ = false;
		return;
	}
	for(int idx = 0; idx < _area.size(); idx = idx+4){
		illegal_parking_area_.push_back({grid_map::Position(_area.at(idx+0), _area.at(idx+1)), grid_map::Position(_area.at(idx+2), _area.at(idx+3))});
	}
	ROS_INFO_STREAM("user_defined_abnormal : ILLEGAL PARKING ON (MIN CAR SPEED : " << _min_car_speed <<")");
}
void osr::user_defined_abnormal::publish_abnormals()
{
	osr_msgs::Abnormals abs_msg;
	abs_msg.header.frame_id = map_frame_id_;
	abs_msg.header.stamp = curr_time_;
	for(int idx = 0; idx < abnormals_.size(); ++idx)
	{
		osr_msgs::Abnormal ab_msg;
		if(abnormals_.at(idx).type == ABNORMAL_TYPE::ILLEGAL_PARKING){
			ab_msg.status = static_cast<int8_t>(1);
			ab_msg.type = static_cast<int8_t>(2);
			ab_msg.detail = static_cast<int8_t>(1);
		}
		else if(abnormals_.at(idx).type == ABNORMAL_TYPE::HIGH_ELEVATION){
			ab_msg.status = static_cast<int8_t>(1);
			ab_msg.type = static_cast<int8_t>(3);
			ab_msg.detail = static_cast<int8_t>(1);
		}
		else if(abnormals_.at(idx).type == ABNORMAL_TYPE::HIGH_THERMAL){
			ab_msg.status = static_cast<int8_t>(1);
			ab_msg.type = static_cast<int8_t>(4);
			ab_msg.detail = static_cast<int8_t>(1);
		}
		else{
			ROS_ERROR("osr_user_defined_abnormal: NOT DEFINED ABNORMALS");
		}
		ab_msg.agent_id = static_cast<int8_t>(abnormals_.at(idx).agent_id);
		ab_msg.pos_x = static_cast<int16_t>(abnormals_.at(idx).pos(0));
		ab_msg.pos_y = static_cast<int16_t>(abnormals_.at(idx).pos(1));
		abs_msg.abnormals.push_back(ab_msg);
	}
	pub_abnormal_.publish(abs_msg);
}
void osr::user_defined_abnormal::set_high_thermal_on(float _max_temp, std::vector<float> _area)
{
	max_temp_ = _max_temp;
	high_thermal_on_ = true;
	if(_area.size() < 4){ // x%4 != 0
		ROS_ERROR("user_defined_abnormal : NOT ENOUGH AREA POINTS FOR HIGH THERMAL.");
		high_thermal_on_ = false;
		return;
	}
	for(int idx = 0; idx < _area.size(); idx = idx+4){
		high_thermal_area_.push_back({grid_map::Position(_area.at(idx+0), _area.at(idx+1)), grid_map::Position(_area.at(idx+2), _area.at(idx+3))});
	}

	ROS_INFO_STREAM("user_defined_abnormal : HIGH THERMAL ON (MAX TEMPERATURE : " << _max_temp <<")");
}
void osr::user_defined_abnormal::set_high_elevation_on(float _max_height, std::vector<float> _area)
{
	max_height_ = _max_height;
	high_elevation_on_ = true;
	if(_area.size() < 4){
		ROS_ERROR("user_defined_abnormal : NOT ENOUGH AREA POINTS FOR HIGH ELEVATION.");
		high_elevation_on_ = false;
		return;
	}
	for(int idx = 0; idx < _area.size(); idx = idx+4){
		high_elevation_area_.push_back({grid_map::Position(_area.at(idx+0), _area.at(idx+1)), grid_map::Position(_area.at(idx+2), _area.at(idx+3))});
	}
	ROS_INFO_STREAM("user_defined_abnormal : HIGH ELEVATION ON (MAX HEIGHT : " << _max_height <<")");
}
void osr::user_defined_abnormal::detect_abnormals()
{
	if(!is_in_map_ || !is_in_agent_info_){
		ROS_ERROR("user_defined_abnormal : NO MAP AND AGENT INFO INPUT.");
		return;
	}

	grid_map::GridMap map;
	grid_map::GridMapRosConverter::fromMessage(map_msg_, map);
	std::map<int, uint64_t> agent_stamps;
	for(int idx = 0; idx < agent_info_msg_.agent_infos.size(); ++idx){
		agent_stamps.insert({agent_info_msg_.agent_infos.at(idx).id, agent_info_msg_.agent_infos.at(idx).timestamp});
	}
	if(illegal_parking_on_){
		for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)		{
			if(map.isValid(*it, "object_update") && map.isValid(*it, "car_number")){
				if(map.at("object_update", *it) > 0){
					float x_vel = map.at("car_flow_x", *it);
					float y_vel = map.at("car_flow_y", *it);
					int agent_id = map.at("object_update", *it);
					if(std::sqrt(x_vel * x_vel + y_vel * y_vel) < min_car_speed_){
						grid_map::Position pos;
						map.getPosition(*it, pos);
						for(int idx = 0; idx < illegal_parking_area_.size(); ++idx){
							if(illegal_parking_area_.at(idx).first(0) > pos(0) && pos(0) > illegal_parking_area_.at(idx).second(0)
									&& illegal_parking_area_.at(idx).first(1) > pos(1) && pos(1) > illegal_parking_area_.at(idx).second(1))
							{
//								abnormals_.push_back({ABNORMAL_TYPE::ILLEGAL_PARKING, pos});
								osr::ABNORMAL ab;
								ab.type = ABNORMAL_TYPE::ILLEGAL_PARKING;
								ab.pos = pos;
								ab.agent_id = agent_id;
								ab.timestamp = agent_stamps[agent_id];
								abnormals_.push_back(ab);
							}
						}
					}
				}
			}
		}
	}
	if(high_thermal_on_){
		for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it){
			if(map.isValid(*it, "thermal_update") && map.isValid(*it, "thermal")){
				if(map.at("thermal_update", *it) > 0){
					int agent_id = map.at("thermal_update", *it);
					if ( map.at("thermal", *it) > max_temp_){
						grid_map::Position pos;
						map.getPosition(*it, pos);
						for(int idx = 0; idx < high_thermal_area_.size(); ++idx){
							if(high_thermal_area_.at(idx).first(0) > pos(0) && pos(0) > high_thermal_area_.at(idx).second(0)
									&& high_thermal_area_.at(idx).first(1) > pos(1) && pos(1) > high_thermal_area_.at(idx).second(1))
							{
//								abnormals_.push_back({ABNORMAL_TYPE::HIGH_THERMAL, pos});
								osr::ABNORMAL ab;
								ab.type = ABNORMAL_TYPE::HIGH_THERMAL;
								ab.pos = pos;
								ab.agent_id = agent_id;
								ab.timestamp = agent_stamps[agent_id];
								abnormals_.push_back(ab);
							}
						}
					}
				}
			}
		}
	}

	if(high_elevation_on_){
		for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it){
			if(map.isValid(*it, "elevation_update") && map.isValid(*it, "elevation")){
				if(map.at("elevation_update", *it) > 0){
					int agent_id = map.at("elevation_update", *it);
					if ( map.at("elevation", *it) > max_height_){
						grid_map::Position pos;
						map.getPosition(*it, pos);
						for(int idx = 0; idx < high_elevation_area_.size(); ++idx){
							if(high_elevation_area_.at(idx).first(0) > pos(0) && pos(0) > high_elevation_area_.at(idx).second(0)
									&& high_elevation_area_.at(idx).first(1) > pos(1) && pos(1) > high_elevation_area_.at(idx).second(1))
							{
//								abnormals_.push_back({ABNORMAL_TYPE::HIGH_ELEVATION, pos});
								osr::ABNORMAL ab;
								ab.type = ABNORMAL_TYPE::HIGH_ELEVATION;
								ab.pos = pos;
								ab.agent_id = agent_id;
								ab.timestamp = agent_stamps[agent_id];
								abnormals_.push_back(ab);
//								int num_abnormal_pxl = 0;
//								for(int ridx = (*it)(0) - 3; ridx < (*it)(0)+3; ++ridx){
//									for(int cidx = (*it)(1) - 3; cidx < (*it)(1)+3; ++cidx)
//									{
//										grid_map::Index pidx(ridx, cidx);
//										if(map.at("elevation", pidx) > max_height_){
//											num_abnormal_pxl++;
//										}
//									}
//								}
//								if(num_abnormal_pxl > 6){
//									abnormals_.push_back({ABNORMAL_TYPE::HIGH_ELEVATION, pos});
//								}
							}
						}
					}
				}
			}

		}

	}
}
void osr::user_defined_abnormal::visualize()
{
	visualization_msgs::MarkerArray mkrarr;
	for (int idx = 0; idx < abnormals_.size(); idx++)
	{
		visualization_msgs::Marker marker;
		marker.header.frame_id = map_frame_id_.c_str();
		marker.header.stamp = curr_time_;
		marker.ns = "pos";
		marker.id = idx;
		marker.type = visualization_msgs::Marker::CYLINDER;
		marker.action = visualization_msgs::Marker::MODIFY;

		marker.pose.position.x = (double) abnormals_.at(idx).pos(0);
		marker.pose.position.y = (double) abnormals_.at(idx).pos(1);
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;

		marker.scale.x = 5.0;
		marker.scale.y = 5.0;
		marker.scale.z = 1.0;

		visualization_msgs::Marker text_marker;
		text_marker.header.frame_id = map_frame_id_.c_str();
		text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
		text_marker.action = visualization_msgs::Marker::MODIFY;
		text_marker.ns = "txt";
		text_marker.header.stamp = curr_time_;
		text_marker.id = idx;

		text_marker.pose.position.x = (double) abnormals_.at(idx).pos(0);
		text_marker.pose.position.y = (double) abnormals_.at(idx).pos(1);
		text_marker.pose.position.z = (double) 2;
		text_marker.pose.orientation.x = 0.0;
		text_marker.pose.orientation.y = 0.0;
		text_marker.pose.orientation.z = 0.0;
		text_marker.pose.orientation.w = 1.0;

		text_marker.scale.z = 1;
		text_marker.frame_locked = true;
		text_marker.lifetime = ros::Duration(20.0);

		if(abnormals_.at(idx).type == ABNORMAL_TYPE::ILLEGAL_PARKING){
			marker.color.a = text_marker.color.a = 1.0; // Don't forget to set the alpha!
			marker.color.r = text_marker.color.r = 1.0;
			marker.color.g = text_marker.color.g = 0.0;
			marker.color.b = text_marker.color.b = 0.0;
			text_marker.text = "ILLEGAL PARKING";
		}
		else if(abnormals_.at(idx).type == ABNORMAL_TYPE::HIGH_THERMAL){
			marker.color.a = text_marker.color.a = 1.0; // Don't forget to set the alpha!
			marker.color.r = text_marker.color.r = 1.0;
			marker.color.g = text_marker.color.g = 0.0;
			marker.color.b = text_marker.color.b = 0.0;
			text_marker.text = "HIGH THERMAL";
		}
		else if(abnormals_.at(idx).type == ABNORMAL_TYPE::HIGH_ELEVATION){
			marker.color.a = text_marker.color.a = 1.0; // Don't forget to set the alpha!
			marker.color.r = text_marker.color.r = 1.0;
			marker.color.g = text_marker.color.g = 0.0;
			marker.color.b = text_marker.color.b = 0.0;
			text_marker.text = "HIGH ELEVATION";
		}
		marker.lifetime = ros::Duration(20.0);
		mkrarr.markers.push_back(marker);
		mkrarr.markers.push_back(text_marker);
	}
	// ZONE marker
	for(int idx = 0; idx < illegal_parking_area_.size(); ++idx){

		visualization_msgs::Marker marker;
		marker.header.frame_id = map_frame_id_.c_str();
		marker.header.stamp = curr_time_;
		marker.ns = "illegal_parking_zone";
		marker.id = idx;
		marker.type = visualization_msgs::Marker::CUBE;
		marker.action = visualization_msgs::Marker::MODIFY;
		float pos_x = (illegal_parking_area_.at(idx).first(0) + illegal_parking_area_.at(idx).second(0))/2.0;
		float pos_y = (illegal_parking_area_.at(idx).first(1) + illegal_parking_area_.at(idx).second(1))/2.0;
		float len_x = (illegal_parking_area_.at(idx).second(0) - illegal_parking_area_.at(idx).first(0));
		float len_y = (illegal_parking_area_.at(idx).second(1) - illegal_parking_area_.at(idx).first(1));
		marker.pose.position.x = (double) pos_x;
		marker.pose.position.y = (double) pos_y;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = len_x;
		marker.scale.y = len_y;
		marker.scale.z = 0.1;
		marker.color.a = 0.3; // Don't forget to set the alpha!
		marker.color.r = 0.3;
		marker.color.g = 0.3;
		marker.color.b = 0.3;

		marker.lifetime = ros::Duration(5.0);
		mkrarr.markers.push_back(marker);
	}
	for(int idx = 0; idx < high_thermal_area_.size(); ++idx){
		visualization_msgs::Marker marker;
		marker.header.frame_id = map_frame_id_.c_str();
		marker.header.stamp = curr_time_;
		marker.ns = "high_thermal_zone";
		marker.id = idx;
		marker.type = visualization_msgs::Marker::CUBE;
		marker.action = visualization_msgs::Marker::MODIFY;
		float pos_x = (high_thermal_area_.at(idx).first(0) + high_thermal_area_.at(idx).second(0))/2.0;
		float pos_y = (high_thermal_area_.at(idx).first(1) + high_thermal_area_.at(idx).second(1))/2.0;
		float len_x = (high_thermal_area_.at(idx).second(0) - high_thermal_area_.at(idx).first(0));
		float len_y = (high_thermal_area_.at(idx).second(1) - high_thermal_area_.at(idx).first(1));
		marker.pose.position.x = (double) pos_x;
		marker.pose.position.y = (double) pos_y;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = len_x;
		marker.scale.y = len_y;
		marker.scale.z = 0.1;
		marker.color.a = 0.3; // Don't forget to set the alpha!
		marker.color.r = 0.3;
		marker.color.g = 0.3;
		marker.color.b = 0.3;

		marker.lifetime = ros::Duration(5.0);
		mkrarr.markers.push_back(marker);
	}
	for(int idx = 0; idx < high_elevation_area_.size(); ++idx){
		visualization_msgs::Marker marker;
		marker.header.frame_id = map_frame_id_.c_str();
		marker.header.stamp = curr_time_;
		marker.ns = "high_elevation_zone";
		marker.id = idx;
		marker.type = visualization_msgs::Marker::CUBE;
		marker.action = visualization_msgs::Marker::MODIFY;
		float pos_x = (high_elevation_area_.at(idx).first(0) + high_elevation_area_.at(idx).second(0))/2.0;
		float pos_y = (high_elevation_area_.at(idx).first(1) + high_elevation_area_.at(idx).second(1))/2.0;
		float len_x = (high_elevation_area_.at(idx).second(0) - high_elevation_area_.at(idx).first(0));
		float len_y = (high_elevation_area_.at(idx).second(1) - high_elevation_area_.at(idx).first(1));
		marker.pose.position.x = (double) pos_x;
		marker.pose.position.y = (double) pos_y;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = len_x;
		marker.scale.y = len_y;
		marker.scale.z = 0.1;
		marker.color.a = 0.3; // Don't forget to set the alpha!
		marker.color.r = 0.3;
		marker.color.g = 0.3;
		marker.color.b = 0.3;

		marker.lifetime = ros::Duration(5.0);
		mkrarr.markers.push_back(marker);
	}


	pub_viz_.publish(mkrarr);

}
void osr::user_defined_abnormal::subscribe_map_callback(const grid_map_msgs::GridMap::ConstPtr& _msg) {
	is_in_map_ = true;
	map_msg_ = *_msg;
}
void osr::user_defined_abnormal::subscribe_agent_info_callback(const osr_msgs::AgentInfos::ConstPtr& _msg) {
	is_in_agent_info_ = true;
	agent_info_msg_ = *_msg;
}
