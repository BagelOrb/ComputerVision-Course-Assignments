/*
 * VoxelTracker.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "General.h"
#include "VoxelTracker.h"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <iostream>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Voxel tracking by clustering
 */
VoxelTracker::VoxelTracker(const vector<Camera*> &cs) :
		_cameras(cs)
{
	for (size_t c = 0; c < _cameras.size(); ++c)
	{
		if (_plane_size.area() > 0)
			assert(_plane_size.width == _cameras[c]->getSize().width && _plane_size.height == _cameras[c]->getSize().height);
		else
			_plane_size = _cameras[c]->getSize();
	}

	initialize();
}

/**
 * Free the memory of the pointer vectors
 */
VoxelTracker::~VoxelTracker()
{
	for (size_t c = 0; c < _clusters.size(); ++c)
		delete _clusters.at(c);
}

/**
 * Creates some bogus clusters for testing
 */
void VoxelTracker::initialize()
{

	Cluster* cluster0 = new Cluster;
	cluster0->center_x = 0;
	cluster0->center_y = 0;
	cluster0->drawColorR = 1.0;
	cluster0->drawColorG = 0.0;
	cluster0->drawColorB = 0.5;

	_clusters.push_back(cluster0);

	Cluster* cluster1 = new Cluster;
	cluster1->center_x = 1000;
	cluster1->center_y = 1000;
	cluster1->drawColorR = 0.5;
	cluster1->drawColorG = 1.0;
	cluster1->drawColorB = 0.0;

	_clusters.push_back(cluster1);
}

/**
 * Count the amount of camera's each cluster in the space appears on,
 * if that amount equals the amount of cameras, add that cluster to the
 * visible_clusters vector
 *
 * Optimized by inverting the process (iterate over clusters instead of camera pixels for each camera)
 */
void VoxelTracker::update()
{
	
}

} /* namespace nl_uu_science_gmt */
