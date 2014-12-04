#include "EuclideanColorModel.h"

#define _USE_MATH_DEFINES
#include <math.h>

Point3d DoubleConeColorModel::getPos(uchar h, uchar l, uchar s)
{
	double x, y, z;

	z = l * height;

	double radAngle = h / 255. * 2 * M_PI;

	double radius = s * min(l+0, 255 - l) / 255.;

	x = cos(radAngle) * radius;
	y = sin(radAngle) * radius;

	return Point3d(x, y, z);
}

static inline double square(double d) { return d*d;  }

uchar DoubleConeColorModel::getDistance(uchar h1, uchar l1, uchar s1, uchar h2, uchar l2, uchar s2)
{
	Point3d p1 = getPos(h1, l1, s1);
	Point3d p2 = getPos(h2, l2, s2);

	return static_cast<uchar>( sqrt(square(p1.x - p2.x) + square(p1.y - p2.y) + square(p1.z - p2.z)) );
}


uchar DoubleConeColorModel::getManhattanDistance(uchar h1, uchar l1, uchar s1, uchar h2, uchar l2, uchar s2)
{
	Point3d p1 = getPos(h1, l1, s1);
	Point3d p2 = getPos(h2, l2, s2);

	return static_cast<uchar>( abs(p1.x - p2.x) + abs(p1.y - p2.y) + abs(p1.z - p2.z) );
}











uchar HLSconditionalColorDistance::getDistance(Vec3b hls1, Vec3b hls2)
{
	double h1 = hls1[0] / 255.;
	double s1 = hls1[1] / 255.;
	double l1 = hls1[2] / 255.;
	double h2 = hls2[0] / 255.;
	double s2 = hls2[1] / 255.;
	double l2 = hls2[2] / 255.;
	double abs_dh = min(abs(h1 - h2), 1.- abs(h1 - h2)); // hue-wrap-around
	double abs_dl = abs(l1 - l2);
	double abs_ds = abs(s1 - s2);
	double l_modifier = (min(l1 * (1 - l1), l2 * (1 - l2)) * 4); // value between 0 and 1, indicating in what respect hue and saturation play a role in the determination of the color of fg or bg
	double s_modifier = min(s1, s2);// value between 0 and 1, indicating in what respect hue plays a role in the determination of the color of fg or bg

	double dist = l_modifier * (s_modifier * abs_dh * weight_h * 4 + (1-s_modifier) * abs_ds * weight_s) + (1. - l_modifier) * abs_dl * weight_l;
	// dh *2 since 0 < dh < .5

	uchar ret = static_cast<uchar>(255 * dist);
	//cout << static_cast<int>(ret) << ", ";
	return ret;
}
