#pragma once
#include <opencv2/opencv.hpp>
#include <math.h> // exp

#include "Scene3DRenderer.h"
#include "HSV_Threshold.h"

using namespace std;



class HSV_Evaluator_Test 
{
public:
	double evaluate(HSV_State& params)
	{
		double h, s, v;
		h = static_cast<double>(params.h);
		s = static_cast<double>(params.s);
		v = static_cast<double>(params.v);
		return exp((h * (100.0 - h) + s * (100.0 - s) + v * (100.0 - v)) / 10000.0);
	};
};

struct HSV_Search_Test
{
	static int main_hsvSearch_test();

};

class HSV_Evaluator 
{

	cv::Mat optimal752;
	cv::Mat background;
	cv::Mat frame752;

public:

	//! Reads the optimal background subtraction result and stores it as a binary matrix
	HSV_Evaluator() {
		cv::Mat optimalimage = cv::imread("data/cam1frame752-manualforeground.bmp");

		vector<cv::Mat> channels;
		cv::split(optimalimage, channels);  // Split the HSV-channels for further analysis

		cv::threshold(channels[0], optimal752, 1, 255, CV_THRESH_BINARY);

		frame752 = cv::imread("data/cam1frame752-original.jpg");
		background = cv::imread("data/background.bmp");
	}

	double evaluate(HSV_State& params)
	{
		
		cv::Mat computed_foreground;

		nl_uu_science_gmt::Scene3DRenderer::processForegroundImproved(frame752, background, computed_foreground, params);

		cv::Mat differenceMatrix, computed_foreground_int; // = cv::Mat::zeros(optimal752.rows, optimal752.cols, CV_32SC1);
			
		//optimal752.copyTo(differenceMatrix);
		optimal752.convertTo(differenceMatrix, CV_32SC1);

		computed_foreground.convertTo(computed_foreground_int, CV_32SC1);

		differenceMatrix -= computed_foreground_int; //!< A matrix containing 0 for matching pixels, 1 for pixels that should have been 1 but are 0 (person classified as background), and -1 for pixels that should have been 0 but are 1 (background classified as person)

		//! Penalty for background classified as person, higher is more bad
		const int false_person = 1;

		//! Penalty for person classified as background, higher is more bad
		const int false_background = 3;
		

		for (int i = 0; i < differenceMatrix.rows; i++) {
			for (int j = 0; j < differenceMatrix.cols; j++)
			{

				int val = differenceMatrix.at<int>(i, j);

				//! Apply weighted penalties to the two different ways in which a pixel can differ
				if (val < 0) { //!< background classified as person
					val = false_person;
					differenceMatrix.at<int>(i, j) = val;

				}
				else if (val > 0) { //!< person classified as background
					val = false_background;
					differenceMatrix.at<int>(i, j) = val;
				}
			}
		}

		//! Add up all the penalties
		auto distance = cv::sum(differenceMatrix)[0];

		return (1000.0 / distance);
	};
};


struct HSV_Search
{
	static int main_hsvSearch();

};