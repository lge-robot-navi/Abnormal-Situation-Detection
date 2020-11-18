/*
 * packets.h
 *
 *  Created on: Apr 14, 2020
 *      Author: osrfix
 */

#ifndef OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_PACKETS_H_
#define OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_PACKETS_H_


#include <iostream>
#include <vector>
#include <string>

namespace osr{

struct abnormal_packet
{
	uint8_t id;
	int report_id;
	float pos_x;
	float pos_y;
	uint8_t status;
	uint8_t type;
	uint8_t detail;
	uint8_t score;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & id;
		ar & report_id;
		ar & pos_x;
		ar & pos_y;
		ar & status;
		ar & type;
		ar & detail;
		ar & score;
	}

};
struct abnormals_packet
{

	void clear(){
		abnormals.erase(abnormals.begin(), abnormals.end());
	}
	uint8_t id;
	std::string agent_id;
	std::vector<abnormal_packet> abnormals;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & id;
		ar & agent_id;
		ar & abnormals;
	}
};

/**
 *  @struct 		image_packet
 *  @brief 			This structure is the message format for image.
 *  @since			Mar 22, 2020
 */
struct image_packet
{


	/**
	* @fn		clear
	* @brief	clear message
	*/
	void clear(){
		data.erase(data.begin(), data.end());
	}
	uint8_t id;
	std::string agent_id;
	float position_x;
	float position_y;
	float rotation_z;
	int rows, cols, channels;
	std::vector<uint8_t> data;

	/**
	* @fn		serialize
	* @brief	serialize member variables to the buffer archive
	* @param 	ar			archive
	* @param	version		version
	*/
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & id;
		ar & agent_id;
		ar & position_x;
		ar & position_y;
		ar & rotation_z;
		ar & rows;
		ar & cols;
		ar & channels;
		ar & data;
	}
};
}


#endif /* OSR_MAP_OSR_OSR_COMM_INCLUDE_OSR_COMM_PACKETS_H_ */
