/**
 * @file		comp_comm.h
 * @date		Mar 22, 2019
 * @author  	Kiin Na (kina4147@etri.re.kr)
 * @brief   	This file includes udp_sync_receiver, udp_sync_sender, udp_receiver, and udp_sender class
 * 				to communicate between modules through UDP. This file supports both synchronized and asynchronized communication.
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

#ifndef COMP_COMM_H_
#define COMP_COMM_H_

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <thread>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/serialization/vector.hpp>



#include "comp_packets.h"
const int MAX_STORAGE_SIZE = 5;
const int MAX_BUFFER_SIZE = 60000;
using boost::asio::ip::udp;
using boost::asio::ip::address;

namespace osr_map{


/**
 *  @class 			udp_sync_receiver
 *  @brief 			This class is for receiving synchronized UDP message
 *  @since			Mar 22, 2019
 */
class udp_sync_receiver {
public:
	udp_sync_receiver(boost::asio::io_service& _io_service, const unsigned int& _port, const int _storage_size = MAX_STORAGE_SIZE) :
		socket_(_io_service, udp::endpoint(udp::v4(), _port)), storage_size_(_storage_size), is_receiving_(false), is_taking_(false){};
	/**
	* @fn		receive_loop
	* @brief	periodically run receive function.
	*/
	void receive_loop(){
		while(1){
			float byte_transferred = receive();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	/**
	* @fn		receive
	* @brief	receive compressed layered_map
	* @return 	bytes_transferred	the size of transferred message
	*/
	size_t receive(){
		try {
			boost::asio::streambuf::mutable_buffers_type mutable_buffer = buffer_.prepare(MAX_BUFFER_SIZE);
			size_t bytes_transferred = socket_.receive(mutable_buffer);
			boost::system::error_code error;
			if (error && error != boost::asio::error::message_size)
				throw boost::system::system_error(error);
			buffer_.commit(bytes_transferred);
			std::istream is(&buffer_);
			boost::archive::binary_iarchive ia(is);
			osr_map::light_layered_map msg;
			ia >> msg;

			if(is_taking_){
				std::cerr << "Layered map is taking..." << std::endl;
				msg.clear();
				return 0;
			}
			is_receiving_ = true;
			if(messages_.size() > storage_size_){
				messages_.front().clear();
				messages_.pop_front();
			}
			messages_.push_back(msg);
			is_receiving_ = false;
			msg.clear();
			return bytes_transferred;
		} catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	}
	/**
	* @fn		get_all_messages
	* @brief	get layered maps
	* @return 	success or fail
	*/
	bool get_all_messages(std::vector<osr_map::light_layered_map>& _msgs)
	{
		if(messages_.size() == 0){return false;}
		std::list<osr_map::light_layered_map> tmp_msgs;
		while(1){
			is_taking_ = true;
			std::cerr << "Layered Map is receiving..." << std::endl;
			if(!is_receiving_){
				_msgs.resize(messages_.size());
				tmp_msgs = messages_;
				for(std::list<osr_map::light_layered_map>::iterator it = messages_.begin(); it != messages_.end(); ++it){it->clear();}
				messages_.erase(messages_.begin(), messages_.end());
				is_taking_ = false;
				break;
			}
		}
		_msgs.resize(tmp_msgs.size());
		std::copy(tmp_msgs.begin(), tmp_msgs.end(), _msgs.begin());
		for(std::list<osr_map::light_layered_map>::iterator it = tmp_msgs.begin(); it != tmp_msgs.end(); ++it){it->clear();}
		tmp_msgs.erase(tmp_msgs.begin(), tmp_msgs.end());
		return true;
	}
	/**
	* @fn		get_new_message
	* @brief	get up-to-date layered maps
	* @return 	msg		layered map
	*/
	osr_map::light_layered_map get_new_message()
	{
		osr_map::light_layered_map msg = messages_.back();
		messages_.clear();
		return msg;
	}
private:
	udp::socket socket_;
	std::list<osr_map::light_layered_map> messages_;
	boost::asio::streambuf buffer_;
	bool is_receiving_;
	bool is_taking_;
	int storage_size_;
};



/**
 *  @class 			udp_sync_sender
 *  @brief 			This class is for sending synchronized UDP message
 *  @since			Mar 22, 2019
 */
class udp_sync_sender {
public:
	udp_sync_sender(boost::asio::io_service& _io_service, const std::string& _ip_address, const unsigned int& _port) :
		socket_(_io_service), receiver_endpoint_(address::from_string(_ip_address), _port){
			socket_.open(udp::v4());
	};
	~udp_sync_sender(){
		socket_.close();
	};
	/**
	* @fn		send
	* @brief	send layered maps
	* @return 	bytes_transferred	the size of transferred message
	*/
	size_t send(const osr_map::light_layered_map & _msg){

		try{
			boost::asio::streambuf buffer;
			std::ostream os(&buffer);
			boost::archive::binary_oarchive ar(os);
			ar << _msg;
			size_t bytes_transferred = socket_.send_to(buffer.data(), receiver_endpoint_);
			return bytes_transferred;
		}
		catch (std::exception& e){
			std::cerr << e.what() << std::endl;
		}
	}
private:
	udp::socket socket_;
	udp::endpoint receiver_endpoint_;
	osr_map::light_layered_map message;
};

/**
 *  @class 			udp_sync_receiver
 *  @brief 			This class is for receiving asynchronized UDP message
 *  @since			Mar 22, 2019
 */
class udp_receiver {
public:
	udp_receiver(boost::asio::io_service& _io_service, const unsigned int& _port, const int _storage_size = MAX_STORAGE_SIZE) :
	socket_(_io_service, udp::endpoint(udp::v4(), _port)), storage_size_(_storage_size){}
	~udp_receiver(){
		socket_.close();
	}

	/**
	* @fn		start_receive
	* @brief	start to wait for receiving compressed layered_map
	* @param 	_size	the maximum buffer size of transferred message
	*/
	void start_receive(std::size_t _size = MAX_BUFFER_SIZE){
		try {
			udp::endpoint sender_endpoint;
			boost::asio::streambuf::mutable_buffers_type mutable_buffer =
					buffer_.prepare(MAX_BUFFER_SIZE);

			socket_.async_receive_from(mutable_buffer,
					sender_endpoint,
					boost::bind(&udp_receiver::handle_receive, this,
							boost::asio::placeholders::error,
							boost::asio::placeholders::bytes_transferred));
		} catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	}

	/**
	* @fn		handle_receive
	* @brief	receive compressed layered_map
	* @param 	error				error message
	* @param 	bytes_transferred	the size of transferred message
	*/
	void handle_receive(const boost::system::error_code& error,	std::size_t bytes_transferred) {
		if (!error || error == boost::asio::error::message_size) {
			buffer_.commit(bytes_transferred);
		    std::istream is(&buffer_);
		    boost::archive::binary_iarchive ia(is);
		    osr_map::light_layered_map msg;
			ia >> msg;
			if(messages_.size() > storage_size_)
				messages_.pop_front();
			messages_.push_back(msg);
			start_receive(bytes_transferred);
		}
	}

	/**
	* @fn		get_all_messages
	* @brief	get layered maps
	* @return 	success or fail
	*/
	bool get_all_messages(std::vector<osr_map::light_layered_map>& _msgs)
	{
		if(messages_.size() == 0)
			return false;
		_msgs.resize(messages_.size());
		std::copy(messages_.begin(), messages_.end(), _msgs.begin());
		messages_.clear();
		return true;
	}


	/**
	* @fn		get_new_message
	* @brief	get up-to-date layered maps
	* @return 	msg		layered map
	*/
	osr_map::light_layered_map get_new_message()
	{
		osr_map::light_layered_map msg = messages_.back();
		messages_.clear();
		return msg;
	}
private:
	udp::socket socket_;
	std::list<osr_map::light_layered_map> messages_;
	boost::asio::streambuf buffer_;
	int storage_size_;
};


/**
 *  @class 			udp_sync_sender
 *  @brief 			This class is for sending asynchronized UDP message
 *  @since			Mar 22, 2019
 */
class udp_sender {
public:
	udp_sender(boost::asio::io_service& _io_service, const std::string& _ip_address, const unsigned int& _port) :
	socket_(_io_service), receiver_endpoint_(address::from_string(_ip_address), _port){
		socket_.open(udp::v4());
	};
	~udp_sender(){
		socket_.close();
	}
	/**
	* @fn		send
	* @brief	send layered maps
	* @return 	bytes_transferred	the size of transferred message
	*/
	size_t send(const osr_map::light_layered_map& _msg)
	{
		try{
			boost::asio::streambuf buffer;
			std::ostream os(&buffer);
			boost::archive::binary_oarchive ar(os);
			ar << _msg;
			boost::system::error_code ignored_error;
			size_t len = socket_.send_to(buffer.data(), receiver_endpoint_, 0, ignored_error);
			buffer.consume(len);
			return len;
		}
		catch (std::exception& e){
			std::cerr << e.what() << std::endl;
		}
	}
private:
	udp::socket socket_;
	udp::endpoint receiver_endpoint_;
	osr_map::light_layered_map message;
};
}

#endif /* COMM_H_ */
