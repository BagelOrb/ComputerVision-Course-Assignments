
#include "stdafx.h"

#include "Main.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> // antialiased line

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



void Asgn1::putTextAt(Mat img, Point3f loc, Scalar color, string text)
{
	vector<Point3f> x; x.push_back(loc);
	vector<Point2f> imagePoints;
	projectPoints(x, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imagePoints);
	putText(img, text, imagePoints[0] + Point2f(-10, 10), FONT_HERSHEY_TRIPLEX, 1, color, 1, 8);
}


void Asgn1::drawApproximatedLine(Mat img, Point3f start, Point3f end, int numberOfSegments, Scalar colour, int thickness)
{
	vector<Point3f> objectPoints;
	objectPoints.push_back(start);
	objectPoints.push_back(end);
	vector<Point2f> imagePoints;
	vector<Point3f> distortedObjectPoints;
	distortedObjectPoints.push_back(objectPoints[0]);
	for (int seg = 1; seg <= numberOfSegments; seg++)
	{
		distortedObjectPoints.push_back(((objectPoints[1] - objectPoints[0]) * (seg / numberOfSegments)) + objectPoints[0]);
	}

	projectPoints(distortedObjectPoints, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imagePoints);

	for (int i = 1; i < imagePoints.size(); i++)
	{
		line(img, imagePoints[i - 1], imagePoints[i], colour, thickness);

	}

}


void Asgn1::drawCube(Mat img, float s)
{	
	int thickness = 1;
	Scalar clr(0, 255, 255);
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(s, 0, 0), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(0, s, 0), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(0, 0, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, 0, s), Point3f(s, 0, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, 0, s), Point3f(0, s, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(s, 0, 0), Point3f(s, 0, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(s, 0, 0), Point3f(s, s, 0), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(s, 0, s), Point3f(s, s, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(s, s, 0), Point3f(s, s, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, s, 0), Point3f(s, s, 0), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, s, s), Point3f(s, s, s), 10, clr, thickness);
	drawApproximatedLine(img, Point3f(0, s, 0), Point3f(0, s, s), 10, clr, thickness);
}

void Asgn1::drawBasis(Mat img, float s)
{	
	int thickness = 2;
	putTextAt(img, Point3f(s, 0, 0), Scalar(255, 0, 0), "X");
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(s, 0, 0), 10, Scalar(255, 0, 0), thickness);

	putTextAt(img, Point3f(0, s, 0), Scalar(0, 255, 0), "Y");
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(0, s, 0), 10, Scalar(0, 255, 0), thickness);

	putTextAt(img, Point3f(0, 0, s), Scalar(0, 0, 255), "Z");
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(0, 0, s), 10, Scalar(0, 0, 255), thickness);

}


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

	drawChessboardCorners(img, Size(6, 9), Mat(corners), found);

	Mat intrinsics, distortion;

	vector<vector<Point2f>> imagePoints;
	imagePoints.push_back(corners);

	vector<vector<Point3f>> realityPoints;
	realityPoints.push_back(Asgn1::getChessboardPoints(Size(6, 9), 3.0));


	calibrateCamera(realityPoints, imagePoints, img.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

	drawBasis(img, 20);
	//drawCube(img, 7);


	return true;
}

VideoCapture cap(0);

string windowName = "Chess or Checkers?";

void Asgn1::capImg(char* file)
{

	if (!boost::filesystem::exists(file))
	{
		cout << "File doesn't exist!" << endl;
		return;
	}


	namedWindow(windowName, WINDOW_AUTOSIZE);

	//processImage(img);
	Mat img = imread(file);// "photo.png");
	processImage(img);
	imshow(windowName, img);

	if (!img.empty())
	{
		waitKey(0);
	}
}

bool startVideo = false; 
bool savePic = false;
int pictureName = 0;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		startVideo = !startVideo; //  !Asgn1::startVideo;
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		savePic = true;
	}
}

void Asgn1::capVideo()
{

	namedWindow(windowName, WINDOW_AUTOSIZE);

	setMouseCallback(windowName, CallBackFunc, NULL);

	cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('B', 'G', 'R', '3'));

	Mat img;

	while (1)
	{

		cap >> img;


		if (!img.empty())
		{
			if (startVideo) processImage(img);
			else putText(img, "Click to start calibration", Point(0, 30), FONT_HERSHEY_TRIPLEX, 1,
				Scalar::all(255), 1, 8);
			imshow(windowName, img);
			if (savePic){
				string save = "../" + to_string(pictureName) + ".jpg";
				imwrite(save, img);
				pictureName++;
				cout << "img saved!" << endl;
				savePic = false;
			}
		}

		if (waitKey(30) >= 0) break;
	}
}


void main(int argc, char** argv)
{
	Asgn1 ass;
	if (argc <= 1)
		ass.capImg("C:\\Users\\TK\\Documents\\Computer Vision\\ComputerVision-Course-Assignments\\Assignment1\\Debug\\board.png");
	//ass.capImg("C:\\Users\\TK\\Documents\\Computer Vision\\Assignments\\ComputerVision-Course-Assignments\\Assignment1\\Debug\\board.png");
	//Asgn1::capImg("C:\\Users\\Marinus\\Documents\\Computer Vision\\Assignments\\ComputerVision-Course-Assignments\\data\\photo.png");
		//cout << "use argument -v to use the standard video capture, and -f [filename] to process a single image" << endl;
	else if (strcmp(argv[1], "-v") == 0)
		ass.capVideo();
	else if (strcmp(argv[1], "-f") == 0)
		ass.capImg(argv[2]);// argv[1]);
	else
	{
		cout << "incorrect argument \"" << argv[1] << "\"" << endl;
		cout << "use argument -v to use the standard video capture, and -f [filename] to process a single image" << endl;
	}
}