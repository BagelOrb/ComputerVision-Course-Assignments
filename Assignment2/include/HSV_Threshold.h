#pragma once

#include <opencv2/opencv.hpp>

#include <chrono>
#include <random>

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
		HSV_State* ret = new HSV_State(uchar(h + dist(gen)), uchar(s + dist(gen)), uchar(v + dist(gen)));


		return ret;
	};


};


