
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


vector<Point3f> getChessboardPoints(Size size)			// function that gets the size of a chessboard and returns a vector of Point3fs where the coordinates are saved.
{
	vector<Point3f> vectorPoint;						// initialize Object vectorPoint of type vector<Point3f>
	int count = 0;										// counter
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			vectorPoint[count] = Point3f(i, j, 0);		// write coordinates in each point
			count++;
		}
	}
	return vectorPoint;									// returns vectorPoint
}

bool processImage(Mat img)
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

	vector<vector<Point2f>> realityPoints;
	realityPoints.push_back(getChessboardPoints(Size(6, 9), 3.0));


	Mat cameraMatrix;
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	calibrateCamera(realityPoints, imagePoints, img.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
	




	return true;
}

VideoCapture cap(0);
Mat frame;


void capImg(char* file)
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

void capVideo()
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
	capImg("C:\\Users\\TK\\Documents\\Computer Vision\\ComputerVision-Course-Assignments\\Assignment1\\Debug\\photo.png");// argv[1]);

}