/*
 * parsers.h
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#ifndef OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_PARSERS_H_
#define OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_PARSERS_H_


#include <opencv2/opencv.hpp>
#include "osr_comm/packets.h"
#include "osr_msgs/Abnormal.h"
namespace osr{
/**
* @fn		convert_light_map_from_grid_map
* @brief	convert grid map to layered map message
* @param	_id					agent id
* @param	_agent_id			agent name
* @param	_seq				sequence number
* @param	_grid_map			grid_map to convert to message
* @param	_min_max_table		minimum and maximum thresholds for layer to compress data
* @param	_map 				layered map message
*/
bool abnormals_to_packet(const int _id, const std::string _agent_id, std::vector<osr_msgs::Abnormal> _abnormals, abnormals_packet& _dst_packet){
	_dst_packet.id = _id;
	_dst_packet.agent_id = _agent_id;
	int count = 0;
	int max_count = 10;
	for (std::vector<osr_msgs::Abnormal>::reverse_iterator rit = _abnormals.rbegin(); rit != _abnormals.rend(); ++rit){
		if(count > max_count){break;}
		osr::abnormal_packet pkt;
		pkt.id = static_cast<uint8_t>(rit->agent_id);
		pkt.report_id = static_cast<int>(rit->report_id);
		pkt.pos_x = static_cast<float>(rit->pos_x);
		pkt.pos_y = static_cast<float>(rit->pos_y);
		pkt.status = static_cast<uint8_t>(rit->status);
		pkt.type = static_cast<uint8_t>(rit->type);
		pkt.detail = static_cast<uint8_t>(rit->detail);
		pkt.score = static_cast<uint8_t>(rit->score);
		_dst_packet.abnormals.push_back(pkt);
		count++;
	}
}
bool packet_to_abnormals(const abnormals_packet& _src_packet, int& _id, std::string& _agent_id, std::vector<osr_msgs::Abnormal>& _abnormals){
	_id = _src_packet.id;
	_agent_id = _src_packet.agent_id;
	for (std::vector<osr::abnormal_packet>::const_iterator it = _src_packet.abnormals.begin(); it != _src_packet.abnormals.end(); ++it){
		osr_msgs::Abnormal msg;
		msg.agent_id = it->id;
		msg.report_id = it->report_id;
		msg.pos_x = it->pos_x;
		msg.pos_y = it->pos_y;
		msg.status = it->status;
		msg.type = it->type;
		msg.detail = it->detail;
		msg.score = it->score;
		_abnormals.push_back(msg);
	}
}
int index_to_order(int _row, int _col, int _rows, int _cols, int _channels){
	return _row * _cols * _channels + _col * _channels;
}
bool image_to_packet(const int _id, const std::string _agent_id, const float _position_x, const float _position_y, const float _rotation_z, const cv::Mat& _src_img, image_packet& _dst_packet){
	_dst_packet.id = static_cast<uint8_t>(_id);
	_dst_packet.agent_id = _agent_id;
	_dst_packet.position_x = _position_x;
	_dst_packet.position_y = _position_y;
	_dst_packet.rotation_z = _rotation_z;
	_dst_packet.rows = _src_img.rows;
	_dst_packet.cols = _src_img.cols;
	_dst_packet.channels = _src_img.channels();
	_dst_packet.data = std::vector<uint8_t>(_dst_packet.rows * _dst_packet.cols * _dst_packet.channels, 0);

	for(int row = 0; row < _src_img.rows; ++row){
		for(int col = 0; col < _src_img.cols; ++col){
			int order = index_to_order(row, col, _src_img.rows, _src_img.cols, _src_img.channels());
			_dst_packet.data[order + 0] = _src_img.at<cv::Vec3b>(row, col)[0];
			_dst_packet.data[order + 1] = _src_img.at<cv::Vec3b>(row, col)[1];
			_dst_packet.data[order + 2] = _src_img.at<cv::Vec3b>(row, col)[2];
		}
	}
}
bool packet_to_image(const image_packet& _src_packet, int& _id, std::string& _agent_id, float& _position_x, float& _position_y, float& _rotation_z, cv::Mat& _dst_img)
{
	_id = static_cast<int>(_src_packet.id);
	_agent_id = _src_packet.agent_id;
	_position_x = _src_packet.position_x;
	_position_y = _src_packet.position_y;
	_rotation_z = _src_packet.rotation_z;
	_dst_img = cv::Mat(_src_packet.rows, _src_packet.cols, CV_8UC3);
	for(int row = 0; row < _src_packet.rows; ++row){
		for(int col = 0; col < _src_packet.cols; ++col){
			int order = index_to_order(row, col, _src_packet.rows, _src_packet.cols, _src_packet.channels);
			_dst_img.at<cv::Vec3b>(row, col)[0] = _src_packet.data[order + 0];
			_dst_img.at<cv::Vec3b>(row, col)[1] = _src_packet.data[order + 1];
			_dst_img.at<cv::Vec3b>(row, col)[2] = _src_packet.data[order + 2];
		}
	}
}
}


#endif /* OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_PARSERS_H_ */
