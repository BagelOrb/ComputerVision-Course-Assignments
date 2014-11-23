
#include "stdafx.h"

#include "Main.h"


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



vector<Point3f> Asgn1::getChessboardPoints(Size size, double gridDistance)
{
	vector<Point3f> vectorPoint;										// initialize Object vectorPoint of type vector<Point3f>
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			vectorPoint.push_back(Point3f(i * gridDistance, j * gridDistance, 0));	// write coordinates in each point
		}
	}
	return vectorPoint;													// returns vectorPoint
}

bool Asgn1::processImage(Mat img)
{
	vector<Point2f> corners; //this will be filled by the detected corners
	bool found = findChessboardCorners(img, Size(6, 9), corners, CV_CALIB_CB_ADAPTIVE_THRESH);
	if (!found) return false;
	
	//for (Point2f p : corners)
	//	circle(img, p, 2, Scalar(255., 0, 0));

	drawChessboardCorners(img, Size(6,9), Mat(corners), found);

	Mat intrinsics, distortion;

	vector<vector<Point2f>> imagePoints;
	imagePoints.push_back(corners);

	vector<vector<Point3f>> realityPoints;
	realityPoints.push_back(Asgn1::getChessboardPoints(Size(6, 9), 3.0));


	Mat cameraMatrix;
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	calibrateCamera(realityPoints, imagePoints, img.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
	




	return true;
}

VideoCapture cap(0);
Mat frame;


void Asgn1::capImg(char* file)
{

	namedWindow("imag", WINDOW_AUTOSIZE);

	//processImage(img);
	Mat img = imread(file);// "photo.png");
	processImage(img);
	imshow("imag", img);

	if (!img.empty())
	{
		cout << "yay!" << endl;
		waitKey(0);
	}
	else cout << ":C" << endl;
}

void Asgn1::capVideo()
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
			//GaussianBlur(frame, frame, Size(17, 17), 15, 15);
			//Canny(frame, frame, 0, 30, 3);
			processImage(frame);
			imshow("imag", frame);

		}
		

		if (waitKey(30) >= 0) break;
	}
}


void main(int argc, char** argv)
{
	//capImg("C:\\Users\\TK\\Documents\\Computer Vision\\ComputerVision-Course-Assignments\\Assignment1\\Debug\\photo.png");// argv[1]);
	//capImg("C:\\Users\\Marinus\\Documents\\Computer Vision\\\Assignments\\ComputerVision-Course-Assignments\\Assignment1\\Debug\\photo.png");// argv[1]);
	Asgn1::capVideo();
}