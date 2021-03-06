/*
Marinus Burger F132726
Tim Kuipers F141459
*/

#include "stdafx.h"

#include "Main.h"


#include <opencv2/core/core.hpp> // antialiased line

#include <boost/filesystem.hpp> // check if image file exists

//  +---------------------------------------------------+
//  | Most functions are explained in the header file,  |
//  | though some global functions are explained below. |
//  +---------------------------------------------------+

// \begin[copied code from example]
#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef __linux__
#include <GL/glut.h>
#include <GL/glu.h>
#endif
// \end[copied code from example]

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <windows.h>
#include <stdio.h>

using namespace cv;
using namespace std;



string windowName = "Chess or Checkers?"; //!< the name and title of the window displayed

bool startVideo = false; //!< a flag used for communication between the mouse callback and Asgn1::capVideo

bool savePic = false; //!< a flag used for communication between the mouse callback and Asgn1::capVideo
int pictureName = 0; //!< a counter to get a semi-unique file name (not unique-unique, cause it will overwrite pics from the preveous session)

/*!
A callback function used to make the video capture respond to mouse clicks.
Toggle chessboard recognition with a simple click!
*/
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


void Asgn1::putTextAt(Mat img, Point3f loc, Scalar color, string text)
{
	vector<Point3f> x; x.push_back(loc);
	vector<Point2f> imagePoints;
	projectPoints(x, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imagePoints);
	putText(img, text, imagePoints[0] + Point2f(-10, 10), FONT_HERSHEY_TRIPLEX, 1, color, 2, 8);
}


void Asgn1::drawApproximatedLine(Mat img, Point3f start, Point3f end, int numberOfSegments, Scalar colour, int thickness)
{
	vector<Point2f> imagePoints;
	vector<Point3f> distortedObjectPoints;
	for (int seg = 0; seg <= numberOfSegments; seg++)
	{
		distortedObjectPoints.push_back(((end - start) * (double(seg) / double(numberOfSegments))) + start);
	}

	projectPoints(distortedObjectPoints, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imagePoints); // project 3D points into image plane

	for (int i = 1; i < imagePoints.size(); i++)
	{
		line(img, imagePoints[i - 1], imagePoints[i], colour, thickness);
	}

}

void Asgn1::drawCube(Mat img, float s, int thickness)
{	
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
void Asgn1::drawBasis(Mat img, float s, int thickness)
{	
	putTextAt(img, Point3f(s, 0, 0), Scalar(200, 50, 50), "X");
	putTextAt(img, Point3f(0, s, 0), Scalar(50, 200, 50), "Y");
	putTextAt(img, Point3f(0, 0, s), Scalar(50, 50, 200), "Z");
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(s, 0, 0), 10, Scalar(255, 0, 0), thickness);
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(0, s, 0), 10, Scalar(0, 255, 0), thickness);
	drawApproximatedLine(img, Point3f(0, 0, 0), Point3f(0, 0, s), 10, Scalar(0, 0, 255), thickness);

}


vector<Point3f> Asgn1::getChessboardPoints(Size size, double gridDistance)
{
	vector<Point3f> vectorPoint;	
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			vectorPoint.push_back(Point3f(i * gridDistance, j * gridDistance, 0));	
		}
	}
	return vectorPoint;	
}

bool Asgn1::processImage(Mat img)
{
	vector<Point2f> corners; //this will be filled by the detected corners
	bool found = findChessboardCorners(img, Size(6, 8), corners, CV_CALIB_CB_ADAPTIVE_THRESH);
	if (!found) return false;

	for (Point2f p : corners)
		circle(img, p, 2, Scalar(255., 255., 0), 2, CV_AA);
	//drawChessboardCorners(img, Size(6, 9), Mat(corners), found);

	Mat intrinsics, distortion;

	vector<vector<Point2f>> imagePoints;
	imagePoints.push_back(corners);

	vector<vector<Point3f>> realityPoints;
	realityPoints.push_back(Asgn1::getChessboardPoints(Size(6, 8), 11.5));


	calibrateCamera(realityPoints, imagePoints, img.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

	ofstream intrinsics1("intrinsics1a_new.txt");
	if (intrinsics1.is_open())
	{
		intrinsics1 << "Camera intrinsics are saved here:\n";
		intrinsics1 << cameraMatrix << "\n";
		intrinsics1 << "-----------------------" << "\n";
		intrinsics1 << distCoeffs << "\n";
		intrinsics1 << "-----------------------" << "\n";
		intrinsics1 << rvecs[0] << "\n";
		intrinsics1 << "-----------------------" << "\n";
		intrinsics1 << tvecs[0] << "\n";
		intrinsics1.close();
	}
	else cout << "Unable to open file";



	drawBasis(img, 20, 2);
	drawCube(img, 7, 3);


	return true;
}


void Asgn1::capImg(char* file)
{

	if (!boost::filesystem::exists(file))
	{
		cout << "File doesn't exist!" << endl;
		return;
	}


	namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat img = imread(file);
	processImage(img);
	imshow(windowName, img);

	if (!img.empty())
	{
		waitKey(0);
	}
}



void Asgn1::capVideo()
{

	namedWindow(windowName, WINDOW_AUTOSIZE);

	setMouseCallback(windowName, CallBackFunc, NULL); // make the window listen to the mouse

	VideoCapture cap(0);

	cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('B', 'G', 'R', '3')); // set teh type of camera

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
		cout << "use argument -v to use the standard video capture, and -f [filename] to process a single image" << endl;
		//ass.capVideo();
		//ass.capImg("C:\\Users\\TK\\Documents\\Computer Vision\\ComputerVision-Course-Assignments\\Assignment1\\Debug\\board_fisheye.png");
	//Asgn1::capImg("C:\\Users\\Marinus\\Documents\\Computer Vision\\Assignments\\ComputerVision-Course-Assignments\\data\\photo.png");
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