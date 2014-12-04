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
	//HSV_Search_Test::main_hsvSearch_test();
	main_(argc, argv);
}
