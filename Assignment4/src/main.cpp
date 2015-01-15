#include <Detector.h>
#include <FeatureHOG.h>

using namespace nl_uu_science_gmt;

int main(int argc, char** argv)
{
	std::string query_image_file = Detector::cfg()->getValue<std::string>("settings/query_image");
	if (argc > 1) query_image_file = (std::string) argv[1];
	std::cout << "Testing on: " << query_image_file << std::endl;

	Detector detector(query_image_file);
	detector.run();

	return EXIT_SUCCESS;
}
