/**
 * @file	user_defined_abnormal.h
 * @date	Mar 22, 2019
 * @author  Kiin Na (kina4147@etri.re.kr)
 * @brief   This file includes user_defined_abnormal detection
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

#ifndef USER_DEFINED_ABNORMAL_H_
#define USER_DEFINED_ABNORMAL_H_

#include <ros/ros.h>
#include <vector>
#include <iostream>
#include <visualization_msgs/MarkerArray.h>
#include "osr_msgs/AgentInfo.h"
#include "osr_msgs/AgentInfos.h"


namespace osr{

class agent_info_viewer{
public:
	agent_info_viewer(ros::NodeHandle _nh, int _period, std::string _map_frame_id, std::string _world_frame_id);
	~agent_info_viewer(){};

	/**
	* @fn            reset
	* @brief         reset member variables
	*
	* This function reset time to estimate the time for a period. It also clear abnormal lists detected at the previous frame.
	*/
	void reset();


	void set_comm(std::string _map_topic, std::string _agent_info_topic, std::string _agent_info_viz_topic);
	/**
	* @fn            visualize
	* @brief         visualize interesting zone for surveilance and abnormals on those position.
	*
	* This function publishes markers for abnormal position and monitoring zones to visualize on RVIZ
	*/
	void visualize();

private:

	/**
	* @fn            subscribe_agent_info_callback
	* @brief         subscribe agent information and save to member variable (agent_info_msg_)
	* @param		 _msg : agent information message
	*/
	void subscribe_agent_info_callback(const osr_msgs::AgentInfos::ConstPtr& _msg);

	ros::NodeHandle nh_;
	ros::Rate period_;
	ros::Time curr_time_, prev_time_;
	std::string map_frame_id_, world_frame_id_;
	osr_msgs::AgentInfos agent_info_msg_;
	ros::Subscriber sub_agent_info_;
	ros::Publisher pub_agent_info_viz_;
	bool is_in_agent_info_;
};

}
#endif /* USER_DEFINED_ABNORMAL_H_ */
