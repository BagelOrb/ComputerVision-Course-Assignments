#include <Detector.h>


#include <FileIO.h>
#include <FeatureHOG.h>
#include <MySVM.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ml/ml.hpp>
#include <Utility.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <utility>



using namespace cv;
using namespace std;
using namespace boost;



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
