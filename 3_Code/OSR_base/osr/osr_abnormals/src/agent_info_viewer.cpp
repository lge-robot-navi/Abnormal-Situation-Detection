/**
 * @file		agent_info_viewer.cpp
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes function definitions that are declared in agent_info_viewer class.
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
#include "osr_abnormals/agent_info_viewer.h"

osr::agent_info_viewer::agent_info_viewer(ros::NodeHandle _nh, int _period, std::string _map_frame_id, std::string _world_frame_id) :
nh_(_nh), period_(_period), map_frame_id_(_map_frame_id), world_frame_id_(_world_frame_id)
{
	curr_time_ = prev_time_ = ros::Time::now();
	is_in_agent_info_ = false;
}

void osr::agent_info_viewer::reset()
{
	curr_time_ = ros::Time::now();
	float time_diff = (float) ((curr_time_ - prev_time_).toNSec() * (1e-6));
	prev_time_ = curr_time_;
}
void osr::agent_info_viewer::set_comm(std::string _map_topic, std::string _agent_info_topic, std::string _agent_info_viz_topic)
{
	sub_agent_info_ = nh_.subscribe(_agent_info_topic, 1, &osr::agent_info_viewer::subscribe_agent_info_callback, this);
	pub_agent_info_viz_ = nh_.advertise < visualization_msgs::MarkerArray > (_agent_info_viz_topic, 1);
}
void osr::agent_info_viewer::visualize()
{
	visualization_msgs::MarkerArray mkrarr;
	int idx = 0;
	for(std::vector<osr_msgs::AgentInfo>::iterator it = agent_info_msg_.agent_infos.begin(); it != agent_info_msg_.agent_infos.end(); ++it)
	{
			visualization_msgs::Marker robot_marker;
			robot_marker.header.frame_id = map_frame_id_.c_str();
			robot_marker.header.stamp = ros::Time::now();
			robot_marker.ns = "pose";
			robot_marker.id = it->id;
			robot_marker.type = visualization_msgs::Marker::CUBE;
			robot_marker.action = visualization_msgs::Marker::MODIFY;

			robot_marker.pose.position.x = (double) it->pos_x;
			robot_marker.pose.position.y = (double) it->pos_y;
			robot_marker.pose.position.z = 0;
			robot_marker.pose.orientation.x = 0.0;
			robot_marker.pose.orientation.y = 0.0;
			robot_marker.pose.orientation.z = 0.0;
			robot_marker.scale.x = 3.0;
			robot_marker.scale.y = 3.0;
			robot_marker.scale.z = 3.0;
			robot_marker.color.a = 1.0; // Don't forget to set the alpha!
			if (it->id > 6){
				robot_marker.color.r = 1.0;
				robot_marker.color.g = 0.0;
				robot_marker.color.b = 0.0;
			}
			else{

				robot_marker.color.r = 0.0;
				robot_marker.color.g = 0.0;
				robot_marker.color.b = 1.0;
			}
			robot_marker.lifetime = ros::Duration(2.0);

			mkrarr.markers.push_back(robot_marker);

			visualization_msgs::Marker text_marker;
			text_marker.header.frame_id = map_frame_id_.c_str();
			text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
			text_marker.action = visualization_msgs::Marker::MODIFY;
			text_marker.ns = "text";
			text_marker.header.stamp = ros::Time::now();
			text_marker.id = idx;

			text_marker.pose.position.x = (double) it->pos_x + 2.0;
			text_marker.pose.position.y = (double) it->pos_y;
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
			text_marker.text = it->id;
			mkrarr.markers.push_back(text_marker);
			idx++;
	}
	pub_agent_info_viz_.publish(mkrarr);

}
void osr::agent_info_viewer::subscribe_agent_info_callback(const osr_msgs::AgentInfos::ConstPtr& _msg) {
	is_in_agent_info_ = true;
	agent_info_msg_ = *_msg;
}
