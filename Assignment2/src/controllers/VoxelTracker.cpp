/*
 * VoxelTracker.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "General.h"
#include "VoxelTracker.h"
#include "VoxelReconstruction.h"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <iostream>

#include <math.h> // isnan

#include <fstream>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

	bool VoxelTracker::RELABEL_EMERGING_VOXELS_ONLY = true; // TK
	bool VoxelTracker::APPLY_CLUSTERING = true; // TK

/**
 * Voxel tracking by clustering
 */
	VoxelTracker::VoxelTracker(Reconstructor &r, const vector<Camera*> &cs, const int numClusters) :
		_reconstructor(r), _cameras(cs)
{
	_num_clusters = numClusters;


	colorModels.load();

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
	
	//Close the text output
	_ftrajectories.close();
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
	cluster0->drawColorB = 0.0;
	cluster0->colorModel = 0;

	_clusters.push_back(cluster0);

	Cluster* cluster1 = new Cluster;
	cluster1->center_x = 1000;
	cluster1->center_y = 0;
	cluster1->drawColorR = 0.0;
	cluster1->drawColorG = 1.0;
	cluster1->drawColorB = 0.0;
	cluster1->colorModel = 0.25;

	_clusters.push_back(cluster1);

	Cluster* cluster2 = new Cluster;
	cluster2->center_x = 2000;
	cluster2->center_y = 0;
	cluster2->drawColorR = 0.0;
	cluster2->drawColorG = 0.0;
	cluster2->drawColorB = 1.0;
	cluster2->colorModel = 0.5;

	_clusters.push_back(cluster2);

	Cluster* cluster3 = new Cluster;
	cluster3->center_x = -1000;
	cluster3->center_y = 1000;
	cluster3->drawColorR = 0.5;
	cluster3->drawColorG = 0.5;
	cluster3->drawColorB = 0.5;
	cluster3->colorModel = 1;

	_clusters.push_back(cluster3);



	//Text output for trajectories
	_ftrajectories.open("output/trajectories.txt", ios::out);
}

/**
 * Update the clusters:
 *
 * For each of the voxels that have become visible, give them a label based on color distance to the nearest label.
 * Recalculate cluster centers based on the average of all the voxels with the cluster label
 * Re-label all voxels based on their distance to these new (final) cluster centers
 */
void VoxelTracker::update()
{

	//Get voxels that are now visible
	std::vector<Reconstructor::Voxel*> currentlyVisibleVoxels = _reconstructor.getVisibleVoxels();

	std::vector<int> numVoxels(_num_clusters); //Number of voxels per cluster
	std::vector<std::vector<int>> clusterVoxelsX(_num_clusters);
	std::vector<std::vector<int>> clusterVoxelsY(_num_clusters);

	//Initialize
	for (size_t c = 0; c < _num_clusters; c++)
	{
		clusterVoxelsX.emplace_back();
		clusterVoxelsY.emplace_back();
	}

	//Check their labels, assign labels based on color distance to those that have no label
	for (size_t i = 0; i < currentlyVisibleVoxels.size(); i++)
	{
		
		//Check if the voxel has a label
		if (!RELABEL_EMERGING_VOXELS_ONLY || currentlyVisibleVoxels[i]->labelNum == -1)
		{

			assignClusterLabelBasedOnColor(currentlyVisibleVoxels[i]);

		}

		//Assign each voxel's coordinates to the voxel's cluster, to calculate cluster centers
		numVoxels[currentlyVisibleVoxels[i]->labelNum]++;
		clusterVoxelsX[currentlyVisibleVoxels[i]->labelNum].push_back(currentlyVisibleVoxels[i]->x);
		clusterVoxelsY[currentlyVisibleVoxels[i]->labelNum].push_back(currentlyVisibleVoxels[i]->y);

	}

	//Recalculate the cluster centers
	for (size_t c = 0; c < _num_clusters; c++)
	{
		if (numVoxels[c] == 0)
		{
			_clusters[c]->center_x = 0;
			_clusters[c]->center_y = 0;
			continue;
		}
		//Sum all the x- and y-coords for this cluster
		int sumx = 0;
		int sumy = 0;
		for (size_t i = 0; i < clusterVoxelsX[c].size(); i++)
		{
			sumx += clusterVoxelsX[c][i];
			sumy += clusterVoxelsY[c][i];
		}

		//Divide by the amount of voxels to obtain mean value
		int meanx = sumx / numVoxels[c];
		int meany = sumy / numVoxels[c];

		//Use the geometrical means to calculate cluster centers which are based more on points closer to the geometrical mean, than on points farther away
		double totweight = 0;
		double totx = 0;
		double toty = 0;
		for (size_t i = 0; i < clusterVoxelsX[c].size(); i++)
		{
			int x = clusterVoxelsX[c][i];
			int y = clusterVoxelsY[c][i];
			double weight = 100 * std::exp(-1e-6 * double( (x - meanx)*(x - meanx) + (y - meany)*(y - meany) ));
			totweight += weight;
			totx += x * weight;
			toty += y * weight;
		}
		if (totweight == 0)
		{
			_clusters[c]->center_x = meanx;
			_clusters[c]->center_y = meany;
		}
		else {
			_clusters[c]->center_x = int(totx / totweight);
			_clusters[c]->center_y = int(toty / totweight);
		}
	}

	if (APPLY_CLUSTERING)
	{
		//Re-label each voxel to the cluster with the minimum spatial (Manhattan) distance from the center to the voxel
		for (size_t i = 0; i < currentlyVisibleVoxels.size(); i++)
		{
			auto minClusterCenterDistance = 100000; // some maximum value
			for (size_t c = 0; c < _num_clusters; c++)
			{
				auto clusterCenterDistance = std::abs(currentlyVisibleVoxels[i]->x - _clusters[c]->center_x) + std::abs(currentlyVisibleVoxels[i]->y - _clusters[c]->center_y);
				if (clusterCenterDistance < minClusterCenterDistance) {
					minClusterCenterDistance = clusterCenterDistance;
					currentlyVisibleVoxels[i]->labelNum = c;
				}
			}
		}
	}

	//Output cluster center data to a file.
	for (size_t c = 0; c < _num_clusters; c++)
	{
		_ftrajectories << "(";
		_ftrajectories << _clusters[c]->center_x << ",";
		_ftrajectories << _clusters[c]->center_y << ") ";

		//Save the cluster positions in the cluster for drawing later
		_clusters[c]->path_x.push_back(_clusters[c]->center_x);
		_clusters[c]->path_y.push_back(_clusters[c]->center_y);
	}
	_ftrajectories << endl;

}

} /* namespace nl_uu_science_gmt */
