#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*!
Geometric model of the normal representation of HSI color space:
A cone from black at the bottom, fully saturated colors on the outer circle in the middle and white at the top.
\beginverbatim
  /\
 /  \
/____\
\    /
 \  /
  \/
\endverbatim

Note that the HSI colorspace coincides with the HLS colorspace, except for its geometric representation.
HLS stands for Hue, Lightness, Saturation.
*/
class DoubleConeColorModel
{
protected:
	double height; //! the height of the double cone model
public:
	//! the only parameter of the double cone color model is the height. The width is set at 1; only the relative ratio between height and width matter.
	DoubleConeColorModel(double height_) : height(height_) {};

	//! get xyz-position in the geometric color space
	Point3d getPos(uchar h, uchar l, uchar s);

	//! Euclidean distance between two colors in hls-format
	uchar getDistance(uchar h1, uchar l1, uchar s1, uchar h2, uchar l2, uchar s2);
	//! Manhattan distance between two colors in hls-format
	uchar getManhattanDistance(uchar h1, uchar l1, uchar s1, uchar h2, uchar l2, uchar s2);
	uchar operator()(Vec3b first, Vec3b second) { return getManhattanDistance(first[0], first[1], first[2], second[0], second[1], second[2]); }; //! Make the class a function object, returning the Manhattan distance between two hls vecs
	//uchar operator()(Vec3b first, Vec3b second) { return getDistance(first[0], first[1], first[2], second[0], second[1], second[2]); }; // Make the class a function object
};

/*!
A color model which maps two colors (in HLS-format) to a distance.
It can be seen as a soft approximation of the following rule:
	- if the lightness is saturated (black or white), ignore hue and saturation.
	- else if either color is a grayscale color, ignore hue.
	- else look at hue, lightness and saturation.

The relative importance of hue, lightness and saturation in the formula are represented by three weights.
*/
class HLSconditionalColorDistance
{
protected:
public:
	float weight_h; //!< the weight for hue
	float weight_l; //!< the weight for lightness
	float weight_s; //!< the weight for saturation

	HLSconditionalColorDistance(uchar wh, uchar wl, uchar ws) 
		: weight_h(static_cast<float>(wh / 255.))
		, weight_l(static_cast<float>(wl / 255.))
		, weight_s(static_cast<float>(ws / 255.)) 
	{};

	uchar getDistance(Vec3b hls1, Vec3b hls2);
	uchar operator()(Vec3b first, Vec3b second) { return getDistance(first, second); }; //! Make the class a function object, mapping two colors to their distance


};
