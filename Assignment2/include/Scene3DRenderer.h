/*
 * Scene3DRenderer.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef SCENE3DRENDERER_H_
#define SCENE3DRENDERER_H_

#include <opencv2/opencv.hpp>
#include <boost/lexical_cast.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <vector>

#include "arcball.h"
#include "General.h"
#include "Reconstructor.h"
#include "VoxelTracker.h"
#include "Camera.h"

#include "HSV_Threshold.h"

namespace nl_uu_science_gmt
{

class Scene3DRenderer
{
public:
	// OURS: (TK
	static bool PERFORM_EROSION_DILATION;// = false;

	enum BackgroundSubtractor { ORIGINAL, CORRECTED, HSL, EUCLIDEAN, CONDITIONAL};
	static BackgroundSubtractor backgroundSubtractor;// = CONDITIONAL;
	// \ours


protected:

	Reconstructor &_reconstructor;
	VoxelTracker &_voxeltracker; //JV
	const std::vector<Camera*> &_cameras;
	const int _num;
	const float _sphere_radius;

	int _square_side_len;
	cv::Size _board_size;

	int _width, _height;
	float _aspect_ratio;

	vec _arcball_eye;
	vec _arcball_centre;
	vec _arcball_up;

	bool _camera_view;
	bool _show_volume;
	bool _show_grd_flr;
	bool _show_cam;
	bool _show_org;
	bool _show_arcball;
	bool _show_info;
	bool _fullscreen;

	bool _quit;
	bool _paused;
	bool _rotate;

	long _number_of_frames;
	int _current_frame;
	int _previous_frame;

	int _current_camera;
	int _previous_camera;

	int _h_threshold;
	int _ph_threshold;
	int _s_threshold;
	int _ps_threshold;
	int _v_threshold;
	int _pv_threshold;

	// edge points of the virtual ground floor grid
	std::vector<std::vector<cv::Point3i*> > _floor_grid;

	void createFloorGrid();

#ifdef _WIN32
	HDC _hDC;
#endif

public:
	//JV: VoxelTracker
	Scene3DRenderer(Reconstructor &, VoxelTracker &, const std::vector<Camera*> &);
	virtual ~Scene3DRenderer();

	void processForeground(Camera*);
	static void processForegroundOriginal(cv::Mat& hsv_image, std::vector<cv::Mat>& bgHsvChannels, cv::Mat& foreground, HSV_State& hsv_thresh);
	static void processForegroundCorrected(cv::Mat& hsv_image, std::vector<cv::Mat>& bgHsvChannels, cv::Mat& foreground, HSV_State& hsv_thresh);
	static void processForegroundHSL(const cv::Mat& bgr_image, std::vector<cv::Mat>& bgHlsChannels, cv::Mat& foreground, HSV_State& hsv_thresh);
	static void processForegroundImproved(const cv::Mat& bgr_image, cv::Mat& bg_image, cv::Mat& foreground, HSV_State& hsv_thresh);
	static void processForegroundImproved2(const cv::Mat& bgr_image, cv::Mat& bg_image, cv::Mat& foreground, HSV_State& hsv_thresh);

	bool processFrame();
	void outputFrame();
	void setCamera(int);
	void setTopView();

	const std::vector<Camera*>& getCameras() const
	{
		return _cameras;
	}

	bool isCameraView() const
	{
		return _camera_view;
	}

	void setCameraView(bool cameraView)
	{
		_camera_view = cameraView;
	}

	int getCurrentCamera() const
	{
		return _current_camera;
	}

	void setCurrentCamera(int currentCamera)
	{
		_current_camera = currentCamera;
	}

	bool isShowArcball() const
	{
		return _show_arcball;
	}

	void setShowArcball(bool showArcball)
	{
		_show_arcball = showArcball;
	}

	bool isShowCam() const
	{
		return _show_cam;
	}

	void setShowCam(bool showCam)
	{
		_show_cam = showCam;
	}

	bool isShowGrdFlr() const
	{
		return _show_grd_flr;
	}

	void setShowGrdFlr(bool showGrdFlr)
	{
		_show_grd_flr = showGrdFlr;
	}

	bool isShowInfo() const
	{
		return _show_info;
	}

	void setShowInfo(bool showInfo)
	{
		_show_info = showInfo;
	}

	bool isShowOrg() const
	{
		return _show_org;
	}

	void setShowOrg(bool showOrg)
	{
		_show_org = showOrg;
	}

	bool isShowVolume() const
	{
		return _show_volume;
	}

	void setShowVolume(bool showVolume)
	{
		_show_volume = showVolume;
	}

	bool isShowFullscreen() const
	{
		return _fullscreen;
	}

	void setShowFullscreen(bool showFullscreen)
	{
		_fullscreen = showFullscreen;
	}

	int getCurrentFrame() const
	{
		return _current_frame;
	}

	void setCurrentFrame(int currentFrame)
	{
		_current_frame = currentFrame;
	}

	bool isPaused() const
	{
		return _paused;
	}

	void setPaused(bool paused)
	{
		_paused = paused;
	}

	bool isRotate() const
	{
		return _rotate;
	}

	void setRotate(bool rotate)
	{
		_rotate = rotate;
	}

	long getNumberOfFrames() const
	{
		return _number_of_frames;
	}

	void setNumberOfFrames(long numberOfFrames)
	{
		_number_of_frames = numberOfFrames;
	}

	bool isQuit() const
	{
		return _quit;
	}

	void setQuit(bool quit)
	{
		_quit = quit;
	}

	int getPreviousFrame() const
	{
		return _previous_frame;
	}

	void setPreviousFrame(int previousFrame)
	{
		_previous_frame = previousFrame;
	}

	int getHeight() const
	{
		return _height;
	}

	int getWidth() const
	{
		return _width;
	}

	cv::Size getSize() const
	{
		return cv::Size(_width, _height);
	}

	void setSize(int w, int h, float a)
	{
		_width = w;
		_height = h;
		_aspect_ratio = a;
	}

	const vec& getArcballCentre() const
	{
		return _arcball_centre;
	}

	const vec& getArcballEye() const
	{
		return _arcball_eye;
	}

	const vec& getArcballUp() const
	{
		return _arcball_up;
	}

	float getSphereRadius() const
	{
		return _sphere_radius;
	}

	float getAspectRatio() const
	{
		return _aspect_ratio;
	}

	const std::vector<std::vector<cv::Point3i*> >& getFloorGrid() const
	{
		return _floor_grid;
	}

	int getNum() const
	{
		return _num;
	}

	Reconstructor& getReconstructor() const
	{
		return _reconstructor;
	}	
	
	//JV
	VoxelTracker& getVoxelTracker() const
	{
		return _voxeltracker;
	}

#ifdef _WIN32
	HDC getHDC() const
	{
		return _hDC;
	}

	void setHDC(const HDC hDC)
	{
		_hDC = hDC;
	}
#endif

	int getPreviousCamera() const
	{
		return _previous_camera;
	}

	int getHThreshold() const
	{
		return _h_threshold;
	}

	int getSThreshold() const
	{
		return _s_threshold;
	}

	int getVThreshold() const
	{
		return _v_threshold;
	}

	int getPHThreshold() const
	{
		return _ph_threshold;
	}

	int getPSThreshold() const
	{
		return _ps_threshold;
	}

	int getPVThreshold() const
	{
		return _pv_threshold;
	}

	void setPHThreshold(int phThreshold)
	{
		_ph_threshold = phThreshold;
	}

	void setPSThreshold(int psThreshold)
	{
		_ps_threshold = psThreshold;
	}

	void setPVThreshold(int pvThreshold)
	{
		_pv_threshold = pvThreshold;
	}

	void setHThreshold(int threshold)
	{
		_h_threshold = threshold;
	}

	void setSThreshold(int threshold)
	{
		_s_threshold = threshold;
	}

	void setVThreshold(int threshold)
	{
		_v_threshold = threshold;
	}

	const cv::Size& getBoardSize() const
	{
		return _board_size;
	}

	int getSquareSideLen() const
	{
		return _square_side_len;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* SCENE3DRENDERER_H_ */
