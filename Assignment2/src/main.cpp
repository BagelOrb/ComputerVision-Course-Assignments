#include "General.h"

#include "VoxelReconstruction.h"

#include <cstdlib>
#include <string>

#include "hsvSearch.h"


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
	Reconstructor::SKIP_VOXELS = true;
	Scene3DRenderer::PERFORM_EROSION_DILATION = false;
	Scene3DRenderer::backgroundSubtractor = Scene3DRenderer::CONDITIONAL;
	//HSV_Search_Test::main_hsvSearch_test(); // TK: test Mixture Model Beam Search
	//HSV_Search::main_hsvSearch(); // TK: find the optimal values for the H S and V slider
	main_(argc, argv);
}
