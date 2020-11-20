/**
 * @file		packets.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes light_map_layer and light_layered_map structure
 * 				that are message format to communicate between agent and server thought UDP communication.
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

#ifndef PACKETS_H_
#define PACKETS_H_

#include <vector>
#include <string>

namespace osr_map{


/**
 *  @struct 		light_map_layer
 *  @brief 			This structure is the message format for map layer.
 *  @since			Mar 22, 2019
 */
struct light_map_layer
{


	/**
	* @fn		clear
	* @brief	clear message
	*/
	void clear(){
		data.erase(data.begin(), data.end());
	}
	std::string name;
	bool normalized;
	float min_val, max_val;
	std::vector<uint8_t> data;

	/**
	* @fn		serialize
	* @brief	serialize member variables to the buffer archive
	* @param 	ar			archive
	* @param	version		version
	*/
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & name;
		ar & min_val;
		ar & max_val;
		ar & data;
	}
};

/**
 *  @struct 		light_layered_map
 *  @brief 			This structure is the message format for layered map.
 *  @since			Mar 22, 2019
 */
struct light_layered_map
{
	/**
	* @fn		clear
	* @brief	clear message
	*/
	void clear(){
		for(std::vector<light_map_layer>::iterator it = map_layers.begin(); it != map_layers.end(); ++it){
			it->clear();
		}
		map_layers.erase(map_layers.begin(), map_layers.end());
//		map_layers.clear();
	}
	std::string agent_id;
	std::string frame_id;
	uint8_t id;
	uint64_t seq;
	uint64_t timestamp;
	float position_x;
	float position_y;
	float length_x;
	float length_y;
	float resolution;
	std::vector<light_map_layer> map_layers;

	/**
	* @fn		serialize
	* @brief	serialize member variables to the buffer archive
	* @param 	ar			archive
	* @param	version		version
	*/
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & agent_id;
		ar & frame_id;
		ar & id;
		ar & seq;
		ar & timestamp;
		ar & position_x;
		ar & position_y;
		ar & length_x;
		ar & length_y;
		ar & resolution;
		ar & map_layers;
	}
};
}


#endif /* PACKETS_H_ */
