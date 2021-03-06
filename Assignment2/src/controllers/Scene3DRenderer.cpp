/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //VideoWriter

#include "Scene3DRenderer.h"
#include "EuclideanColorModel.h" // TK : new background subtraction algorithm
#include "Utils.h" // TK : (TODO remove)

#include <stddef.h>
#include <cassert>
#include <string>
#include <algorithm> // TK:  transform


using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

bool Scene3DRenderer::PERFORM_EROSION_DILATION = false;

Scene3DRenderer::BackgroundSubtractor Scene3DRenderer::backgroundSubtractor = CONDITIONAL;


/**
 * Scene properties class (mostly called by Glut)
 */
//JV: VoxelTracker, VideoWriter
Scene3DRenderer::Scene3DRenderer(Reconstructor &r, VoxelTracker &vt, const vector<Camera*> &cs) :
_reconstructor(r), _voxeltracker(vt), _cameras(cs), _num(4), _sphere_radius(1850)
{
	_width = 640;
	_height = 480;
	_quit = false;
	_paused = true; //JV
	_rotate = false;
	_camera_view = true;
	_show_volume = true;
	_show_grd_flr = true;
	_show_cam = true;
	_show_org = true;
	_show_arcball = false;
	_show_info = true;
	_fullscreen = false;
	_draw_cluster_paths = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> _board_size.width;
		fs["CheckerBoardHeight"] >> _board_size.height;
		fs["CheckerBoardSquareSize"] >> _square_side_len;
	}
	fs.release();

	_current_camera = 0;
	_previous_camera = 0;

	_number_of_frames = _cameras.front()->getFramesAmount();
	_current_frame = 0;
	_previous_frame = -1;

	const int H = 183;// 167; // (different hyperparameter settings)
	const int S = 149;// 138;
	const int V = 90;// 75;
	//const int H = 174;// (different hyperparameter settings)
	//const int S = 130;// 
	//const int V = 65;// 
	_h_threshold = H;
	_ph_threshold = H;
	_s_threshold = S;
	_ps_threshold = S;
	_v_threshold = V;
	_pv_threshold = V;

	createTrackbar("Frame", VIDEO_WINDOW, &_current_frame, _number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &_v_threshold, 255);

	createFloorGrid();
	setTopView();

	//VideoWriter for video output
	_wrt.open("output/video1.avi", 1145656920, 29, Size(640, 480));
}

/**
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < _floor_grid.size(); ++f)
		for (size_t g = 0; g < _floor_grid[f].size(); ++g)
			delete _floor_grid[f][g];

	//Close our video output
	_wrt.release();
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < _cameras.size(); ++c)
	{
		if (_current_frame == _previous_frame + 1)
		{
			_cameras[c]->advanceVideoFrame();
		}
		else if (_current_frame != _previous_frame)
		{
			_cameras[c]->getVideoFrame(_current_frame);
		}
		assert(_cameras[c] != nullptr);
		processForeground(_cameras[c]);
	}
	return true;
}

//JV
/**
* Output whatever is in the gl frame buffer to a file. This is always a frame late.
* Adapted from http://stackoverflow.com/questions/9097756/converting-data-from-glreadpixels-to-opencvmat
*/
void Scene3DRenderer::outputFrame()
{
	int width = getWidth(), height = getHeight();
	cv::Mat img(height, width, CV_8UC3), flipped(height, width, CV_8UC3);

	//use fast 4-byte alignment (default anyway) if possible
	glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);

	//set length of one complete row in destination data (doesn't need to equal img.cols)
	glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());

	glReadPixels(0, 0, img.cols, img.rows, 0x80E0 /* GL_BGR */, GL_UNSIGNED_BYTE, img.data);

	cv::flip(img, flipped, 0);

	//GAH more C++ bullshit
	//std::string filename = "output/3doutput";
	//filename = filename + boost::lexical_cast<std::string>(getCurrentFrame());
	//filename = filename + ".bmp";

	//imwrite(filename, flipped);

	_wrt.write(flipped);
}

/**
* Separate the background from the foreground
* ie.: Create an 8 bit image where only the foreground of the scene is white
*/
void Scene3DRenderer::processForeground(Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> bgHsvChannels = camera->getBgHsvChannels();

	Mat foreground, bg_image;

	bg_image = camera->getBgImage();


	switch (backgroundSubtractor)
	{
	case ORIGINAL:
		processForegroundOriginal(hsv_image, bgHsvChannels, foreground, HSV_State(_h_threshold, _s_threshold, _v_threshold)); 
		break;
	case CORRECTED:
		processForegroundCorrected(hsv_image, bgHsvChannels, foreground, HSV_State(_h_threshold, _s_threshold, _v_threshold));
		break;
	case HSL:
		processForegroundHSL(camera->getFrame(), bgHsvChannels, foreground, HSV_State(_h_threshold, _s_threshold, _v_threshold));
		break;
	case EUCLIDEAN:
		processForegroundImproved2(camera->getFrame(), bg_image, foreground, HSV_State(_h_threshold, _s_threshold, _v_threshold));
		break;
	case CONDITIONAL:
		processForegroundImproved(camera->getFrame(), bg_image, foreground, HSV_State(_h_threshold, _s_threshold, _v_threshold));
		break;
	}


	int size = 2;
	Mat kernel = getStructuringElement(MORPH_RECT,
		Size(2 * size + 1, 2 * size + 1),
		Point(size, size));

	if (PERFORM_EROSION_DILATION)
	{
		erode(foreground, foreground, kernel, Point(-1, -1), 1); // remove white specks
		dilate(foreground, foreground, kernel, Point(-1, -1), 2); // go back and remove black holes
		erode(foreground, foreground, kernel, Point(-1, -1), 1); // go back
	}



	// Improve the foreground image

	camera->setForegroundImage(foreground);
}
void Scene3DRenderer::processForegroundOriginal(Mat& hsv_image, vector<Mat>& bgHsvChannels, Mat& foreground, HSV_State& hsv_thresh)
{

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	// Background subtraction H
	Mat tmp, background;
	absdiff(channels[0], bgHsvChannels.at(0), tmp);
	threshold(tmp, foreground, hsv_thresh.h, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(channels[1], bgHsvChannels.at(1), tmp);
	threshold(tmp, background, hsv_thresh.s, 255, CV_THRESH_BINARY);
	bitwise_and(foreground, background, foreground);

	// Background subtraction V
	absdiff(channels[2], bgHsvChannels.at(2), tmp);
	threshold(tmp, background, hsv_thresh.v, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, background, foreground);

}

void Scene3DRenderer::processForegroundCorrected(Mat& hsv_image, vector<Mat>& bgHsvChannels, Mat& foreground, HSV_State& hsv_thresh)
{

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	Mat tmp, foregroundH1, foregroundH2, foregroundS, foregroundV;
	// Background subtraction H
	absdiff(channels[0], bgHsvChannels.at(0), tmp);
	threshold(tmp, foregroundH1, hsv_thresh.h, 255, CV_THRESH_BINARY);
	threshold(tmp, foregroundH2, 255 - hsv_thresh.h, 255, CV_THRESH_BINARY_INV); //TK
	bitwise_and(foregroundH1, foregroundH2, foreground); //TK : hue-wrap-around

	// Background subtraction S
	absdiff(channels[1], bgHsvChannels.at(1), tmp);
	threshold(tmp, foregroundS, hsv_thresh.s, 255, CV_THRESH_BINARY);
	bitwise_and(foregroundS, foreground, foreground);

	// Background subtraction V
	absdiff(channels[2], bgHsvChannels.at(2), tmp);
	threshold(tmp, foregroundV, hsv_thresh.v, 255, CV_THRESH_BINARY);
	bitwise_or(foregroundV, foreground, foreground);

}

void Scene3DRenderer::processForegroundHSL(const Mat& bgr_image, vector<Mat>& bgHlsChannels, Mat& foreground, HSV_State& hsv_thresh)
{
	Mat hsv_image;
	cvtColor(bgr_image, hsv_image, CV_BGR2HLS);  // TK: from BGR to HLS color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	Mat tmp, foregroundH1, foregroundH2, foregroundS, foregroundL;
	// Background subtraction H
	absdiff(channels[0], bgHlsChannels.at(0), tmp);
	threshold(tmp, foregroundH1, hsv_thresh.h, 255, CV_THRESH_BINARY);
	threshold(tmp, foregroundH2, 255 - hsv_thresh.h, 255, CV_THRESH_BINARY_INV); //TK
	bitwise_and(foregroundH1, foregroundH2, foreground); //TK : hue-wrap-around

	// Background subtraction L
	absdiff(channels[1], bgHlsChannels.at(1), tmp);
	threshold(tmp, foregroundL, hsv_thresh.v, 255, CV_THRESH_BINARY);
	bitwise_and(foregroundL, foreground, foreground);

	// Background subtraction S
	absdiff(channels[2], bgHlsChannels.at(2), tmp);
	threshold(tmp, foregroundS, hsv_thresh.s, 255, CV_THRESH_BINARY);
	bitwise_or(foregroundS, foreground, foreground);


}

void Scene3DRenderer::processForegroundImproved(const Mat& bgr_image, Mat& bg_image, Mat& foreground, HSV_State& hsv_thresh)
{
	Mat hsv_image;
	cvtColor(bgr_image, hsv_image, CV_BGR2HLS);  // TK: from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis




	HLSconditionalColorDistance comp(hsv_thresh.h, hsv_thresh.v, hsv_thresh.s); // h=h, l=v, s=s
	//cout << comp.weight_h << ", " << comp.weight_l << ", " << comp.weight_s << endl;

	Mat bg_hls_im;
	cvtColor(bg_image, bg_hls_im, CV_BGR2HLS);

	Mat dists = channels[0].clone(); // initialize the same type and dimensions as a single color channel mat 

	std::transform(hsv_image.begin<Vec3b>(), hsv_image.end<Vec3b>(), bg_hls_im.begin<Vec3b>(), dists.begin<uchar>(), comp);

	threshold(dists, foreground, 10, 255, CV_THRESH_BINARY);

	//cout << "next"<<endl;


}
void Scene3DRenderer::processForegroundImproved2(const Mat& bgr_image, Mat& bg_image, Mat& foreground, HSV_State& hsv_thresh)
{
	Mat hsv_image;
	cvtColor(bgr_image, hsv_image, CV_BGR2HLS);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis




	DoubleConeColorModel comp(hsv_thresh.h * 4. / 255.);


	Mat  bg_hls_im;
	cvtColor(bg_image, bg_hls_im, CV_BGR2HLS);

	Mat dists = channels[0].clone(); // initialize the same type and dimensions as a single color channel mat 

	std::transform(hsv_image.begin<Vec3b>(), hsv_image.end<Vec3b>(), bg_hls_im.begin<Vec3b>(), dists.begin<uchar>(), comp);

	threshold(dists, foreground, hsv_thresh.s, 255, CV_THRESH_BINARY);

}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(int camera)
{
	_camera_view = true;

	if (_current_camera != camera)
	{
		_previous_camera = _current_camera;
		_current_camera = camera;
		_arcball_eye.x = _cameras[camera]->getCameraPlane()[0].x;
		_arcball_eye.y = _cameras[camera]->getCameraPlane()[0].y;
		_arcball_eye.z = _cameras[camera]->getCameraPlane()[0].z;
		_arcball_up.x = 0.0f;
		_arcball_up.y = 0.0f;
		_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	_camera_view = false;
	if (_current_camera != -1) _previous_camera = _current_camera;
	_current_camera = -1;

	_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = _reconstructor.getSize();
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * _num; y <= size * _num; y += size)
		edge1.push_back(new Point3i(-size * _num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * _num; x <= size * _num; x += size)
		edge2.push_back(new Point3i(x, size * _num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * _num; y <= size * _num; y += size)
		edge3.push_back(new Point3i(size * _num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * _num; x <= size * _num; x += size)
		edge4.push_back(new Point3i(x, -size * _num, z_offset));

	_floor_grid.push_back(edge1);
	_floor_grid.push_back(edge2);
	_floor_grid.push_back(edge3);
	_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
