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

using namespace cv;
using namespace std;

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

	char key;
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
/*
std::shared_ptr<cv::Mat> matrix;

int main(int argc, char** argv)
{
	
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
	{
		cout << "Coulnt open cam" << endl;
		return -1;
	}
	
	bool success = cap.grab();
	cout << success << endl;

	Mat img;

	cap.retrieve(img);
	
	Range colR;
	img.colRange(colR);
	Range rowR;
	img.rowRange(rowR);


	cout << colR.end <<"-"<< colR.start << " x " << rowR.end<<"-"<<rowR.start << endl;

	if (success)
	{
		namedWindow("img", 1);
		while (waitKey(30) < 0)
			imshow("img", img);

	}


	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);
		if (waitKey(30) >= 0) break;
	} 
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
*/