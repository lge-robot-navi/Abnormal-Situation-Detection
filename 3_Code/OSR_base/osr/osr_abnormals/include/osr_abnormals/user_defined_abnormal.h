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
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/GridMap.hpp>
#include <visualization_msgs/MarkerArray.h>
#include "osr_msgs/Abnormal.h"
#include "osr_msgs/Abnormals.h"
#include "osr_msgs/AgentInfo.h"
#include "osr_msgs/AgentInfos.h"


namespace osr{

enum ABNORMAL_TYPE { ILLEGAL_PARKING, HIGH_THERMAL, HIGH_ELEVATION};
/**
 *  @struct 		user_defined_abnormal
 *  @brief 			abnormal structure
 *  @since			Mar 22, 2019
 */
struct ABNORMAL{
	ABNORMAL_TYPE type;
	int agent_id;
	uint64_t timestamp;
	grid_map::Position pos;
};
/**
 *  @class 			user_defined_abnormal
 *  @brief 			This user_defined_abnormal class is to detect abnormals likes illegal parking, high temperature, and high elevation on multi-layered map (including object_update, car_flow_x, car_flow_y, car_number, thermal_update, thermal, elevation_update, elevation)
 *  	   			if all the data (map and agent information) are gathered. Also, it visualize and publish abnormals and monitoring zones on RVIZ
 *  @since			Mar 22, 2019
 */
class user_defined_abnormal{
public:
	user_defined_abnormal(ros::NodeHandle _nh, int _period, std::string _map_frame_id, std::string _world_frame_id);
	~user_defined_abnormal(){};

	/**
	* @fn            reset
	* @brief         reset member variables
	*
	* This function reset time to estimate the time for a period. It also clear abnormal lists detected at the previous frame.
	*/
	void reset();

	/**
	* @fn            detect_abnormals
	* @brief         reset member variables
	*
	* This function detect user defined abnormals like illegal parking, high temperature, and high elevation
	* on multi-layered map (including object_update, car_flow_x, car_flow_y, car_number, thermal_update, thermal, elevation_update, elevation)
	* if all the data (map and agent information) are gathered
	*/
	void detect_abnormals();

	/**
	* @fn            visualize
	* @brief         visualize interesting zone for surveilance and abnormals on those position.
	*
	* This function publishes markers for abnormal position and monitoring zones to visualize on RVIZ
	*/
	void visualize();

	/**
	* @fn            set_comm
	* @brief         set topic name for ROS subscriber and publisher
	* @param 		 _map_topic : map topic name for subscribing
	* 				 _agent_info_topic : agent information topic name for subscribing
	* 				 _abnormal_topic : abnormal topic for publishing
	*
	* This function set the name of essential topics to communicate with other nodes.
	*/
	void set_comm(std::string _map_topic, std::string _agent_info_topic, std::string _abnormal_topic);

	/**
	* @fn            set_viz_on
	* @brief         turn on visualization
	*
	* This function turn on visualization by setting publisher for visualization markers.
	*/
	void set_viz_on();


	/**
	* @fn           set_illegal_parking_on
	* @brief        This function turns on abnormal detection of illegal parking.
	* 				It sets minimum car speed for detecting parked cars and saves monitoring zones for illegal parking.
	* @param 		_min_car_speed : minimum speed to assume parked cars
	* 				_area : monitoring zones for illegal parking
	*
	*/
	void set_illegal_parking_on(float _min_car_speed, std::vector<float> _area);

	/**
	* @fn           set_high_thermal_on
	* @brief        This function turns on abnormal detection of high temperature. It sets maximum temperature
	* 				for detecting high temperature and saves monitoring zones for high temperature.
	* @param 		_max_temp : maximum temperature threshold for user-defined abnormals
	* 				_area : monitoring zones for high temperature
	*
	*/
	void set_high_thermal_on(float _max_temp, std::vector<float> _area);

	/**
	* @fn           set_high_elevation_on
	* @brief        This function turns on abnormal detection of high elevation like strange static objects. It sets maximum elevation and monitoring zones to detect strange object in the certain areas.
	* @param 		_max_height : maximum height threshold for user-defined abnormals
	* 				_area : monitoring zones for high elevation
	*
	*/
	void set_high_elevation_on(float _max_height, std::vector<float> _area);

	/**
	* @fn            publish_abnormals
	* @brief         publish abnormals after transferring to abnormal message
	*
	* This function reset time to estimate the time for a period. It also clear abnormal lists detected at the previous frame.
	*/
	void publish_abnormals();

private:

	/**
	* @fn            subscribe_map_callback
	* @brief         subscribe grid map message and save to member variable (map_msgs_)
	* @param		 _msg : grid map message
	*/
	void subscribe_map_callback(const grid_map_msgs::GridMap::ConstPtr& _msg);

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
	grid_map_msgs::GridMap map_msg_;
	osr_msgs::AgentInfos agent_info_msg_;
	grid_map::GridMap map_;
	ros::Subscriber sub_map_, sub_agent_info_;
	ros::Publisher pub_abnormal_, pub_viz_;
	float min_car_speed_, max_temp_, max_height_;
	bool is_in_map_, is_in_agent_info_;
	bool illegal_parking_on_, high_thermal_on_, high_elevation_on_;
	std::vector<std::pair<grid_map::Position, grid_map::Position>> illegal_parking_area_, high_thermal_area_, high_elevation_area_;
//	std::vector<std::pair<ABNORMAL_TYPE, grid_map::Position>> abnormals_;
	std::vector<ABNORMAL> abnormals_;
};

}
#endif /* USER_DEFINED_ABNORMAL_H_ */
