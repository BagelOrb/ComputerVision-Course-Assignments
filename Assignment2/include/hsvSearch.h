#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
#include <math.h> // exp

using namespace std;

/*
class ParamState
{
public:
	virtual ParamState* getNearbyRandom(double std_dev, std::default_random_engine& gen);


};
*/

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

/*
template<class ParamStateType>
class PerformanceEvaluator_Base
{
public:
	virtual double evaluate(ParamStateType& params);
};
*/

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

class HSV_Evaluator 
{
public:
	double evaluate(HSV_State& params)
	{
		
		return 0;
	};
};
