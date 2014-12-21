/*
* main.cpp
*
*  Created on: 10 Nov 2014
*      Author: coert
*/
#include "stdafx.h"

#include <memory>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef __linux__
#include <GL/glut.h>
#include <GL/glu.h>
#endif



#include <algorithm> // TK:  transform


using namespace cv;

std::shared_ptr<cv::Mat> matrix;

struct Transformer
{
	EM em;
	Transformer(EM em) : em(em) {};

	Vec3b operator()(Vec3b a)
	{
		Vec2d ret = em.predict(a);
		uchar val = ret[0] * 256;
		return Vec3b(val, val, val);
	};

};

int main(int argc, char** argv)
{
	
	cv::Mat me = cv::imread("C:\\Users\\TK\\Documents\\Computer Vision\\ComputerVision-Course-Assignments\\AssignmentZero\\Debug\\Koala.jpg");// argv[1]);
	matrix = std::make_shared<cv::Mat>(me);

	/*
	boost::filesystem::create_directory("data");

	while (true)
	{
		cv::namedWindow("OpenCV", CV_WINDOW_KEEPRATIO);
		cv::imshow("OpenCV", *matrix);
		char key = cv::waitKey(10);
		if (key == 27) exit(EXIT_SUCCESS);
	}

	*/

	EM em(3);

	em.train(me);

	//Mat m(2, 2, CV_8UC3, Scalar(0, 0, 0));

	Mat out(me);

	Transformer transformer(em);

	//std::transform(me.begin<Vec3b>(), me.end<Vec3b>(), out.begin<Vec3b>(), transformer);


	return EXIT_SUCCESS;
}
