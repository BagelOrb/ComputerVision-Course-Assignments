#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
#include <math.h> // exp

using namespace std;


class HSV_State 
{
public:
	uchar h;
	uchar s;
	uchar v;

	HSV_State(uchar h_, uchar s_, uchar v_)
		: h(h_), s(s_), v(v_)
	{};

	HSV_State* getNearbyRandom(double std_dev, std::default_random_engine& gen)
	{
		std::normal_distribution<double> dist(0, std_dev);
		HSV_State* ret = new HSV_State(h + dist(gen), s + dist(gen), v + dist(gen));
		

		return ret;
	};


};


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
public:
	double evaluate(HSV_State& params)
	{
		
		cv::Mat computed_foreground, optimal_foreground;

		cv::Mat differenceMatrix = optimal_foreground - computed_foreground; //!< A matrix containing 0 for matching pixels, 1 for pixels that should have been 1 but are 0 (person classified as background), and -1 for pixels that should have been 0 but are 1 (background classified as person)

		//! Penalty for background classified as person, higher is more bad
		const int false_person = 1;

		//! Penalty for person classified as background, higher is more bad
		const int false_background = 3;
		

		for (int i = 0; i < differenceMatrix.rows; i++) {
			for (int j = 0; j < differenceMatrix.cols; j++)
			{
				int val = differenceMatrix.at<int>(i, j);

				//! Apply weighted penalties to the two different ways in which a pixel can differ
				if (val == -1) { //!< background classified as person
					val = false_person;
					differenceMatrix.at<uchar>(i, j) = val;

				}
				else if (val == 1) { //!< person classified as background
					val = false_background;
					differenceMatrix.at<uchar>(i, j) = val;
				}
			}
		}

		//! Add up all the penalties
		auto distance = cv::sum(differenceMatrix)[0];

		return (1.0 / distance);
	};
};
