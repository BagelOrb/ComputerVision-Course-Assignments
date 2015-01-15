/*
 * MySVM.cpp
 *
 *  Created on: 27 aug. 2013
 *      Author: Coert
 */

#include "MySVM.h"

namespace nl_uu_science_gmt {

MySVM::MySVM() :
	is_trained(false)
{
}

MySVM::~MySVM() {
}

void MySVM::run(const cv::Mat &trainData, const cv::Mat &responses, const cv::SVMParams &params)
{
	is_trained = train(trainData, responses, cv::Mat(), cv::Mat(), params);
}

} /* namespace nl_uu_science_gmt */
