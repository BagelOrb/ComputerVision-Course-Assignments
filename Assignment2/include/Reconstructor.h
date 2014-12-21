/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/opencv.hpp>
#include <stddef.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv2/opencv.hpp>

#include "Camera.h"

#include <stdint.h> // int16_t

namespace nl_uu_science_gmt
{

class Reconstructor
{
	friend class VoxelTracker;
public:
	static bool SKIP_VOXELS; // TK

	struct Voxel
	{
		int16_t x, y, z; // TK: smaller int type; bigger isnt needed

		//The 'label' of the cluster this voxel belongs to, -1 if none JV
		char labelNum;

		std::vector<cv::Point> camera_projection;
		std::vector<int> valid_camera_projection;
	};

private:
	const std::vector<Camera*> &_cameras;

	int _step;
	int _size;

	std::vector<cv::Point3f*> _corners;

	size_t _voxels_amount;
	cv::Size _plane_size;

	std::vector<Voxel*> _voxels;
	std::vector<Voxel*> _visible_voxels;

	void initialize();

public:
	Reconstructor(const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return _visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return _voxels;
	}

	void setVisibleVoxels(const std::vector<Voxel*>& visibleVoxels)
	{
		_visible_voxels = visibleVoxels;
	}

	void setVoxels(const std::vector<Voxel*>& voxels)
	{
		_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return _corners;
	}

	int getSize() const
	{
		return _size;
	}

	const cv::Size& getPlaneSize() const
	{
		return _plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
