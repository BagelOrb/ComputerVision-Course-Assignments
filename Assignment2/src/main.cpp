#include "General.h"

#include "VoxelReconstruction.h"

#include <cstdlib>
#include <string>

#include "hsvSearch.h"
#include "MixtureColorModel.h"


#include "Scene3DRenderer.h"
#include "Reconstructor.h"

#include <algorithm> // transform
#include <opencv2/opencv.hpp>

#include <opencv2/ml/ml.hpp>

#include <boost/filesystem.hpp>
#include <memory>


using namespace cv;
using namespace nl_uu_science_gmt;
int main_(int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
	Reconstructor::SKIP_VOXELS = false;
	Scene3DRenderer::PERFORM_EROSION_DILATION = false;
	Scene3DRenderer::backgroundSubtractor = Scene3DRenderer::CONDITIONAL;

	VoxelTracker::RELABEL_EMERGING_VOXELS_ONLY = false;
	VoxelTracker::APPLY_CLUSTERING = false;

	main_(argc, argv);

	/*
	MixtureColorModel mcm;
	mcm.generateModels();
	mcm.saveModels();
	*/
	
	/*
	MixtureColorModel m;
	m.load();
	

	m.test();
	*/

}



