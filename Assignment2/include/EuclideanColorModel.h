#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class DoubleConeColorModel
{
protected:
	double height;
public:

	DoubleConeColorModel(double height_) : height(height_) {};
	Point3d getPos(uchar h, uchar l, uchar s);
	uchar getDistance(uchar h1, uchar l1, uchar s1, uchar h2, uchar l2, uchar s2);
	uchar getManhattanDistance(uchar h1, uchar l1, uchar s1, uchar h2, uchar l2, uchar s2);
	uchar operator()(Vec3b first, Vec3b second) { return getManhattanDistance(first[0], first[1], first[2], second[0], second[1], second[2]); }; // Make the class a function object
	//uchar operator()(Vec3b first, Vec3b second) { return getDistance(first[0], first[1], first[2], second[0], second[1], second[2]); }; // Make the class a function object
};

class HLSconditionalColorDistance
{
protected:
public:
	float weight_h;
	float weight_l;
	float weight_s;

	HLSconditionalColorDistance(uchar wh, uchar wl, uchar ws) 
		: weight_h(static_cast<float>(wh / 255.))
		, weight_l(static_cast<float>(wl / 255.))
		, weight_s(static_cast<float>(ws / 255.)) 
	{};

	uchar getDistance(Vec3b hls1, Vec3b hls2);
	uchar operator()(Vec3b first, Vec3b second) { return getDistance(first, second); }; // Make the class a function object


};
