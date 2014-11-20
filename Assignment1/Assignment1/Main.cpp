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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <windows.h>
#include <stdio.h>

using namespace cv;
using namespace std;

VideoCapture cap(0);

Mat frame;
void main()
{
	
	namedWindow("imag", WINDOW_AUTOSIZE);
	// Sleep(1000);


	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
	//cap.set(CV_CAP_PROP_FPS, 15);
	cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('B', 'G', 'R', '3'));


	while (1)
	{

		cap >> frame;


		if (!frame.empty())
		{
			GaussianBlur(frame, frame, Size(17, 17), 15, 15);
			Canny(frame, frame, 0, 30, 3);
			imshow("imag", frame);
		}
		if (waitKey(30) >= 0) break;

	}
}