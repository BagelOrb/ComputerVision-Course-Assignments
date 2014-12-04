#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>

using namespace std;

class ParamState
{
public:
	virtual ParamState* getNearbyRandom(double std_dev, std::default_random_engine gen);


};


class HSV_State //: public ParamState
{
public:
	uchar h;
	uchar s;
	uchar v;

	HSV_State(uchar h_, uchar s_, uchar v_)
		: h(h_), s(s_), v(v_)
	{};

	HSV_State* getNearbyRandom(double std_dev, std::default_random_engine gen)
	{
		std::normal_distribution<double> dist(0, std_dev);
		HSV_State* ret = new HSV_State(h + dist(gen), s + dist(gen), v + dist(gen));
		cout << static_cast<int>(ret->h) << ", " << static_cast<int>(ret->s) << ", " << static_cast<int>(ret->v) << endl;

		return ret;
	};


};


template<class ParamStateType>
class PerformanceEvaluator_Base
{
public:
	virtual double evaluate(ParamStateType& params);
};

class HSV_Evaluator //: public PerformanceEvaluator_Base < HSV_State >
{
public:
	double evaluate(HSV_State& params)
	{
		return params.h * (100 - params.h) + params.s * (100 - params.s) + params.v * (100 - params.v);
	};
};
