#pragma once

#include "stdafx.h"

#include <memory>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;

/*!
Assignment 1 of the course Computer Vision:
- Calibrate camera
- Draw the basis of the world coordinate system
- Draw a cube with its lower corner located at the origin

Extra:
- Approximate curved lines of the cube and basis, so that the lines curves as much as the fisheye lens effect
*/
class Asgn1
{
public:
	Mat cameraMatrix;
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;

	//! put text on the image near a point in 3D
	void putTextAt(Mat img, Point3f loc, Scalar color, string text);

	//! draw an approximate curve corresponding toi a straight line in 3D space
	void drawApproximatedLine(Mat img, Point3f start, Point3f end, int numberOfSegments, Scalar colour, int thickness);

	/*! draw the cube which has one vertex at the origin
	\param img the image on which to draw
	\param size the size (in world space) of each side of the cube 
	\param thickness the thickness (2D) of the lines drawn
	*/
	void drawCube(Mat img, float size, int thickness);

	/*! draw the basis of the world coordinate system (and the axis names)
	\param img the image on which to draw
	\param size the length (in world space) of the line to be drawn along each axis
	\param thickness the width in 2D image space of the line drawn
	*/
	void drawBasis(Mat img, float size, int thickness);

	/*!
	Returns a vector of 3D points corresponding to the junctions in a grid of the given size, at Z=0.
	\param size the grid dimensions
	\param gridDistance the distance betwee two neighboring junction points
	*/
	static vector<Point3f> getChessboardPoints(Size size, double gridDistance);
	/*!
	Recognizes a 6x9 chessboard (and displays it) and displays the basis of the coordinate system and a cube located at the origin
	*/
	bool processImage(Mat img);
	/*!
	Perform the assignment for a single image
	*/
	void capImg(char* file);


	/*!
	Perform the assignment for the standard video stream
	*/
	void capVideo();



};