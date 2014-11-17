/*
 * main.cpp
 *
 *  Created on: 10 Nov 2014
 *      Author: coert
 */

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

std::shared_ptr<cv::Mat> matrix;

int main(int argc, char** argv)
{
	cv::Mat me = cv::imread(argv[1]);
	matrix = std::make_shared<cv::Mat>(me);

	boost::filesystem::create_directory("data");

	while(true)
	{
		cv::namedWindow("OpenCV", CV_WINDOW_KEEPRATIO);
		cv::imshow("OpenCV", *matrix);
		char key = cv::waitKey(10);
		if(key == 27) exit(EXIT_SUCCESS);
	}
	
	return EXIT_SUCCESS;
}
