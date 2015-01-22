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
	/*
	const Mat image = imread(query_image_file);

	cv::Mat HOG_features;
	FeatureHOG<float>::compute(image, HOG_features);
	Mat reshaped_HOG_features = HOG_features.reshape(FeatureHOG<float>::DEPTH);


	if (_use_hog)
	{
	cv::Mat HOG_features;
	FeatureHOG<float>::compute(features, HOG_features);
	features = HOG_features.reshape(FeatureHOG<float>::DEPTH);
	}


	std::vector<cv::Mat> feature_slices;
	cv::split(features.reshape(FeatureHOG<float>::DEPTH), feature_slices); // Reshape to DEPTH channels, and split into vector of channel slices
	
	//--operate on feature slices--
	cout << feature_slices.size() << endl;

	Mat resized;
	resize(feature_slices[0], resized, Size(200, 200));
	imshow("feature slice 0", resized);

	Mat viz = FeatureHOG<float>::visualize(features);
	imshow("viz", viz);
	//imshow("all feature slices ?!", feature_slices);

	cout << "image : "<< image.dims << '\t';
	for (int i = 0; i<image.dims; ++i)
		cout << image.size[i] << ',';
	cout << endl;

	cout << "features : "<<features.dims << '\t';
	for (int i = 0; i<features.dims; ++i)
		cout << features.size[i] << ',';
	cout << endl;


	cout << "reshaped : " << reshaped.dims << '\t';
	for (int i = 0; i<reshaped.dims; ++i)
		cout << reshaped.size[i] << ',';
	cout << endl;

	cout << "FeatureHOG<T>::DEPTH = " << FeatureHOG<float>::DEPTH << endl;

	cv::merge(feature_slices, features);                               // Merge into single matrix
	features.reshape(1).copyTo(features);                              // Re-interleave all channel slices
	
	cout << "entering loop" << endl;


	waitKey();
	*/
	
	if (argc > 1) query_image_file = (std::string) argv[1];
	std::cout << "Testing on: " << query_image_file << std::endl;

	Detector detector(query_image_file);
	detector.run();
	

	return EXIT_SUCCESS;
}
