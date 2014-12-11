/*
 * VoxelTracker.h
 *
 *  Created on: Nov 15, 2013
 *      Author: Jeroen
 */

#ifndef VoxelTracker_H_
#define VoxelTracker_H_

#include <opencv2/opencv.hpp>
#include <stddef.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "Reconstructor.h"

namespace nl_uu_science_gmt
{

class VoxelTracker
{
public:

	struct Cluster
	{
		int center_x, center_y;
		//Color to draw this cluster as
		float drawColorR;
		float drawColorG;
		float drawColorB;

		//Probably some other stuff here
		std::vector<Reconstructor::Voxel> closestVoxels;
	};

private:
	const std::vector<Camera*> &_cameras;

	int _step;
	int _size;


	cv::Size _plane_size;

	std::vector<Cluster*> _clusters;

	void initialize();

public:
	VoxelTracker(const std::vector<Camera*> &);
	virtual ~VoxelTracker();

	void update();

	const std::vector<Cluster*>& getClusters() const
	{
		return _clusters;
	}


	void setClusters(const std::vector<Cluster*>& clusters)
	{
		_clusters = clusters;
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

#endif /* VoxelTracker_H_ */
