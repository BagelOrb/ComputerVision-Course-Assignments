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

#include <iostream>
#include <fstream>

//#include <cstdlib> // rand (TK)

#include "Camera.h"
#include "Reconstructor.h"

#include "MixtureColorModel.h"


namespace nl_uu_science_gmt
{

class VoxelTracker
{
protected:
	Reconstructor &_reconstructor;
	MixtureColorModel colorModels;
public:

	std::ofstream _ftrajectories;

	static bool RELABEL_EMERGING_VOXELS_ONLY;	
	static bool APPLY_CLUSTERING;

	struct Cluster
	{
		int center_x, center_y;
		//Color to draw this cluster as
		float drawColorR;
		float drawColorG;
		float drawColorB;

		//A history of the cluster center's path
		std::vector<int> path_x;
		std::vector<int> path_y;

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


	uchar getLabelMinimalColorDistanceCluster(Vec3b color)
	{
		uchar label = 0;
		auto minColorModelDistance = VoxelTracker::colorModelMaxDistance; // some maximum value
		for (uchar m = 0; m < _num_clusters; m++)
		{

			double colorModelDistance = colorModels.distance(color, m);
			if (colorModelDistance < minColorModelDistance) {
				minColorModelDistance = colorModelDistance;
				label = m;
			}
		}
		return label;
	}

	//Assign a label based on majority vote of minimum color distance to each cluster's color model
	void assignClusterLabelBasedOnColor(Reconstructor::Voxel* voxel)
	{
		std::vector<int> votes(_reconstructor._cameras.size());
		votes.resize(_reconstructor._cameras.size(), 0); // fill with zeros (TK)

		for (int cam = 0; cam < _reconstructor._cameras.size(); cam++)
		{
			const Point point = voxel->camera_projection[cam];
			Vec3b color = _cameras[cam]->getFrame().at<Vec3b>(point);
			votes[getLabelMinimalColorDistanceCluster(color)] ++;

		}
		
		std::vector<uchar>* maximalVotes = new std::vector<uchar>();
		uchar maxVotes = 0;
		for (uchar cluster = 0; cluster < _num_clusters; cluster++)
		{
			if (votes[cluster] > maxVotes)
			{
				maxVotes = votes[cluster];
				delete maximalVotes;
				maximalVotes = new std::vector<uchar>();
				maximalVotes->push_back(cluster);
			}
			else if (votes[cluster] == maxVotes)
			{
				maximalVotes->push_back(cluster);
			}
		}

		if (maximalVotes->size() == 1) // one cluster has maximal number of votes
		{
			voxel->labelNum = (*maximalVotes)[0];
			return;
		}
		else // multiple clusters have a tie!
		{
			voxel->labelNum = (*maximalVotes)[rand() % maximalVotes->size()];
			return;
		}



	}

	/*
	//Calculates the 'distance' between a voxels color and a given color model
	double colorDistance(Reconstructor::Voxel* voxel, int colorModel)
	{
		std::vector<int> votes(_reconstructor._cameras.size());
		votes.resize(_reconstructor._cameras.size(), 0); // fill with zeros (TK)

		for (int c = 0; c < _reconstructor._cameras.size(); c++)
		{
			const Point point = voxel->camera_projection[c];
			if (_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;

		}
		//cv::Vec3b color = ;

	}
	*/

	//Some maximum value for the result of colorDistance
	const double colorModelMaxDistance = 1000;

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
