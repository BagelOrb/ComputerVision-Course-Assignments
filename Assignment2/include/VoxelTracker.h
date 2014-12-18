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
protected:
	Reconstructor &_reconstructor;

public:

	struct Cluster
	{
		int center_x, center_y;
		//Color to draw this cluster as
		float drawColorR;
		float drawColorG;
		float drawColorB;

		//Probably some other stuff here
		//std::vector<Reconstructor::Voxel> closestVoxels;

		//Color model: TODO change this to something useful instead of int
		int colorModel;
	};

private:
	const std::vector<Camera*> &_cameras;

	int _num_clusters;

	cv::Size _plane_size;

	std::vector<Cluster*> _clusters;

	void initialize();

public:
	VoxelTracker(Reconstructor &, const std::vector<Camera*> &, const int);
	virtual ~VoxelTracker();

	void update();

	const std::vector<Cluster*>& getClusters() const
	{
		return _clusters;
	}

	//Calculates the 'distance' between a color and a given color model
	float colorDistance(cv::Scalar color, int colorModel)
	{
		return 0; //TODO implement
	}

	//Some maximum value for the result of colorDistance
	const float colorModelMaxDistance = 1000;

	void setClusters(const std::vector<Cluster*>& clusters)
	{
		_clusters = clusters;
	}

	const cv::Size& getPlaneSize() const
	{
		return _plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* VoxelTracker_H_ */
