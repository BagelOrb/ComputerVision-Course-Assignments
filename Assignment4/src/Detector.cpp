/*
 * Detector.cpp
 *
 *  Created on: Aug 23, 2013
 *      Author: coert
 */

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


#define DEBUG_HERE() cout << "line:" << __LINE__ << "\t(debug)" << endl; 
#define DEBUG_SHOW(x) cout << "Detector." << __LINE__ << ": " << #x << " = " << x << endl;

namespace nl_uu_science_gmt
{

#define CONFIG_PATH "data/config.xml"
QueryXML* Detector::_Config = nullptr;
string Detector::StorageExt = "xml";
string Detector::ImageExt = "png";

Detector::Detector(const string &qif) :
		_pos_amount(Detector::cfg()->getValue<int>("settings/images@amount")), _target_width(
				Detector::cfg()->getValue<int>("settings/images@width")), _posneg_factor(
				Detector::cfg()->getValue<int>("settings/images@factor")), _seed(
				Detector::cfg()->getValue<int>("settings@seed")), _disp(
				Detector::cfg()->getValue<int>("settings/images/examples@size")), _max_count(
				Detector::cfg()->getValue<size_t>("settings/svm/params/max_count")), _epsilon(
				Detector::cfg()->getValue<double>("settings/svm/params/epsilon")), _max_image_size(
				Detector::cfg()->getValue<int>("settings/images/test/max_size")), _initial_threshold(
				Detector::cfg()->getValue<int>("settings/images/test@threshold")), _nms_overlap_threshold(
				Detector::cfg()->getValue<double>("settings/images/test/nms@threshold")), _gt_accuracy(
				Detector::cfg()->getValue<double>("settings/images/test/accuracy@threshold")), 
				_eval_everywhere(Detector::cfg()->getValue<bool>("settings/features/hog/eval_everywhere")), // (TK)
				_use_hog(Detector::cfg()->getValue<bool>("settings/features/hog/use_hog")), // (TK)
				_do_equalizing((_use_hog)? false : // (TK) don't do whitening or equalization when using HOG
				Detector::cfg()->getValue<bool>("settings/features@equalize")), 
				_do_whitening((_use_hog)? false : // (TK) don't do whitening or equalization when using HOG
				Detector::cfg()->getValue<bool>("settings/features@whiten")), _show_ground_truth(
				Detector::cfg()->getValue<bool>("settings/images/test@show_ground_truth")), _query_image_file(qif),
				_pyramid_height(Detector::cfg()->getValue<int>("settings/features/pyramid/height")), // (TK)
				_pyramid_downscale_factor(Detector::cfg()->getValue<double>("settings/features/pyramid/downscale_factor")), // (TK)
				_smallestImageModelSizeFactor(Detector::cfg()->getValue<double>("settings/features/pyramid/smallestImageModelSizeFactor")), // (TK)
				_min_slider_value(50), _max_slider_value(100)
{

	assert(_max_count > 0);
	assert(_epsilon > 0);

	if (_seed > 0)
		srand((unsigned int) _seed);
	else
		srand((unsigned int) cvGetTickCount());
}

Detector::~Detector()
{
	cfg_destroy();
}

QueryXML* Detector::cfg()
{
	if (!_Config) // Only allow one instance of class to be generated.
		_Config = new QueryXML(CONFIG_PATH);

	return _Config;
}

void Detector::cfg_destroy()
{
	if (_Config) delete _Config;
	_Config = nullptr;
}

/*
 * Read positive image files (and shuffle)
 */
void Detector::readPosFilelist(vector<string> &pos_files)
{
	const string pos_path = Detector::cfg()->getValue<string>("settings/images/pos/path");
	const string dirs_regx = Detector::cfg()->getValue<string>("settings/images/pos/path@regx");
	const string mask_regx = Detector::cfg()->getValue<string>("settings/images/pos/files@regx");

	cout << "Read positive file names" << endl;
	vector<string> pos_directories;
	FileIO::getDirectory(pos_path, pos_directories, dirs_regx);

	vector<double> fps;
	int index = 0;
	int64 t0 = Utility::get_time_curr_tick();
	for (size_t d = 0; d < pos_directories.size(); ++d)
	{
		string directory = pos_directories[d];
		vector<string> d_files;
		string full_path = pos_path + directory + "/";
		assert(FileIO::isDirectory(full_path));
		FileIO::getDirectory(full_path, d_files, mask_regx, full_path);
		pos_files.insert(pos_files.end(), d_files.begin(), d_files.end());

		string etf = Utility::show_fancy_etf(index, (int) pos_directories.size(), 100, t0, fps);
		if (!etf.empty()) cout << etf << endl;

		++index;
	}

	assert(pos_files.size() > 0);
	random_shuffle(pos_files.begin(), pos_files.end());
}

bool strfindreplace(std::string& str, const std::string& from, const std::string& to) {
	size_t start_pos = str.find(from);
	if (start_pos == std::string::npos)
		return false;
	str.replace(start_pos, from.length(), to);
	return true;
}

/*
 * Read negative image files (and filter, then shuffle)
 */
void Detector::readNegFilelist(vector<string> &neg_files)
{
	cout << "Read negative file names" << endl;
	const string neg_path = Detector::cfg()->getValue<string>("settings/images/neg/path");
	const string files_regx = Detector::cfg()->getValue<string>("settings/images/neg/files@regx");
	FileIO::getDirectory(neg_path, neg_files, files_regx, neg_path);

	cout << "Before: " << neg_files.size() << endl;
	for (int i = 0; i < neg_files.size(); i++)
	{
		ifstream annotationFile;
		string line;
		string filename = neg_files.at(i); 
		
		//JPEGImages -> Annotations, jpg -> xml
		strfindreplace(filename, "JPEGImages", "Annotations");
		strfindreplace(filename, "jpg", "xml");

		//Open the annotation file
		annotationFile.open(filename);

		if (annotationFile.is_open()) {
			while (getline(annotationFile, line)) {

				//See if the image contains a person
				//(This is the quickest way and based on a test it is equivalent to actually parsing the xml)
				if (line.find("person") != std::string::npos) {

					//"person" found, exclude the file
					//cout << "Excluding img because it contains a person: " << neg_files.at(i) << endl;
					neg_files.erase(neg_files.begin()+i);
					--i; //compensate
					break; //no need to read the rest of the file
				}
			}
			annotationFile.close();
		}
		else {
			cerr << "Error opening Annotations file " << filename << endl;
		}
	}

	cout << "After: " << neg_files.size() << endl;

	assert(!neg_files.empty());
	random_shuffle(neg_files.begin(), neg_files.end());
}

/*
 * Read positive image data
 */
void Detector::readPosData(const Strings &pos_train, Mat &pos_data)
{
	assert(!pos_train.empty());

	const int x1 = Detector::cfg()->getValue<int>("settings/features/crop/x1");
	const int y1 = Detector::cfg()->getValue<int>("settings/features/crop/y1");
	const int x2 = Detector::cfg()->getValue<int>("settings/features/crop/x2");
	const int y2 = Detector::cfg()->getValue<int>("settings/features/crop/y2");

	Mat image = imread(pos_train.front(), CV_LOAD_IMAGE_GRAYSCALE);
	const int width = _target_width;
	const int height = int(width * double(image.rows / (double) image.cols));
	if (_model_size.area() == 0) _model_size = Size(width, height);

	cout << "Read positive data (" << pos_train.size() << ")" << endl;
	Mat mv_pos_data, sv_pos_data;
	int index = 0;
	vector<double> fps;
	int64 t0 = Utility::get_time_curr_tick();
	Mat pos_sum;

	for (size_t f = 0; f < pos_train.size(); ++f)
	{
		string file = pos_train[f];
		image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		Rect rect = Rect(Point(x1, y1), Point(image.cols - x2, image.rows - y2));
		assert(rect.area() < image.size().area());
		Mat features = image(rect);
		resize(features, features, _model_size);

		/*
		if (_use_hog)
		{
			cv::Mat HOG_features;
			FeatureHOG<float>::compute(features, HOG_features);
			features = HOG_features.reshape(FeatureHOG<float>::DEPTH);
		}
		*/

		if (pos_sum.empty())
		{
			Mat sz_imF;
			features.convertTo(sz_imF, CV_64F);
			pos_sum.push_back(sz_imF);
		}
		else
		{
			Mat sz_imF;
			features.convertTo(sz_imF, CV_64F);
			pos_sum += sz_imF;
		}

		Mat features1d = features.reshape(1, 1);
		Mat features1dF;
		features1d.convertTo(features1dF, CV_64F);

		Mat mean, stddev;
		meanStdDev(features1dF, mean, stddev);
		Mat mean_line(1, features1dF.cols, CV_64F);
		mean_line = mean.at<double>(0, 0);
		Mat std_line(1, features1dF.cols, CV_64F);
		std_line = stddev.at<double>(0, 0);
		mv_pos_data.push_back(mean_line);
		sv_pos_data.push_back(std_line);


		pos_data.push_back(features1dF);

		string etf = Utility::show_fancy_etf(index, (int) pos_train.size(), 10, t0, fps);
		if (!etf.empty()) cout << etf << endl;

		++index;
	}

	if (_do_equalizing)
	{
		pos_data = pos_data - mv_pos_data;
		pos_data = pos_data / sv_pos_data;
	}

	// This is the mean model

	Mat pos_sum1dF = pos_sum.reshape(1, 1);
	_pos_sumF = pos_sum1dF / pos_data.rows;

	normalize(pos_sum1dF.reshape(1, height), pos_sum, 255, 0, NORM_MINMAX);
	pos_sum.convertTo(_pos_sum8U, CV_8U);

}

/*
 * Read negative image data
 */
void Detector::readNegData(const Strings &neg_train, Mat &neg_data)
{
	assert(!neg_train.empty());

	Mat mv_neg_data, sv_neg_data;

	double factor = _pos_amount * _posneg_factor / (double) neg_train.size();
	int fpnt = (int) ceil(MAX(factor, 1));

	cout << "Read negative data (" << neg_train.size() * fpnt << ")" << endl;
	vector<double> fps;
	int index = 0;
	int64 t0 = Utility::get_time_curr_tick();
	Mat neg_sum;
	for (size_t f = 0; f < neg_train.size(); ++f)
	{
		string file = neg_train[f];
		Mat image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

		for (int i = 0; i < fpnt; ++i)
		{
			int x = 0 + (rand() % ((image.cols - _model_size.width) - 0));
			int y = 0 + (rand() % ((image.rows - _model_size.height) - 0));
			assert(x + _model_size.width < image.cols);
			assert(y + _model_size.height < image.rows);
			Mat features = image(Rect(Point(x, y), _model_size)).clone();

			if (neg_sum.empty())
			{
				Mat sz_imF;
				features.convertTo(sz_imF, CV_64F);
				neg_sum.push_back(sz_imF);
			}
			else
			{
				Mat sz_imF;
				features.convertTo(sz_imF, CV_64F);
				neg_sum += sz_imF;
			}
			/*
			if (_use_hog)
			{
				cv::Mat HOG_features;
				FeatureHOG<float>::compute(features, HOG_features);
				features = HOG_features.reshape(FeatureHOG<float>::DEPTH);
			}
			*/

			Mat features1d = features.reshape(1, 1);
			Mat features1dF;
			features1d.convertTo(features1dF, CV_64F);

			Mat mean_val, stddev_val;
			meanStdDev(features1dF, mean_val, stddev_val);
			Mat mean_line(1, features1dF.cols, CV_64F);
			mean_line = mean_val.at<double>(0, 0);
			Mat std_line(1, features1dF.cols, CV_64F);
			std_line = stddev_val.at<double>(0, 0);
			mv_neg_data.push_back(mean_line);
			sv_neg_data.push_back(std_line);

			neg_data.push_back(features1dF);

			string etf = Utility::show_fancy_etf(index, (int) neg_train.size() * fpnt, _posneg_factor * 10, t0, fps);
			if (!etf.empty()) cout << etf << endl;

			++index;
		}
	}

	if (_do_equalizing)
	{
		neg_data = neg_data - mv_neg_data;
		neg_data = neg_data / sv_neg_data;
	}

	normalize(neg_sum, neg_sum, 255, 0, NORM_MINMAX);
	neg_sum.convertTo(_neg_sum8U, CV_8U);
}

/*! @brief train a Model based on train_data and train_labels
 *
 * @param train_data a 2D matrix containing the training data, 1 example per row
 * @param train_labels a 1D matrix (vector) containing 1 example label per row (1 or -1)
 */
void Detector::train(const Mat &train_data, const Mat &train_labels, Model &model)
{
	cout << endl << "Build SVM model" << endl;

	if(model.svm == nullptr) model.svm = std::make_shared<MySVM>();			// Initialize new SVM if necessary

	const double C = Detector::cfg()->getValue<double>("settings/svm/params/C");
	cout << endl << "line:" << __LINE__ << ") C: " << C << endl;

	Mat data;
	if (train_data.type() != CV_32F)
		train_data.convertTo(data, CV_32F);
	else
		data = train_data;

	Mat labels;
	if (train_labels.type() != CV_32S)
		train_labels.convertTo(labels, CV_32S);
	else
		labels = train_labels;

	// Set up SVM's parameters
	SVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = C;															// Fault tolerance
	params.kernel_type = SVM::POLY;						// We setup SVM as a 1st degree polynomial (result is the same as SVM::LINEAR)
	params.degree = 1;												// polynomial: (gamma*u'*v + coef0)^degree
	params.gamma = 1;
	params.coef0 = 0;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int) _max_count, _epsilon);

	// Train the SVM
	cout << "line:" << __LINE__ << ") Training SVM..." << endl;
	
	Mat HOG_data;
	if (_use_hog)
	{
		Mat HOG_data_8U;
		for (int i = 0; i < train_data.rows; i++)
		{
			Mat row = train_data.row(i);
			Mat window = row.reshape(1, _model_size.height);
			Mat window_8U;
			window.convertTo(window_8U, CV_8U);
			cv::Mat HOG_features;
			FeatureHOG<float>::compute(window_8U, HOG_features);
			Mat reshaped = HOG_features.reshape(1, 1);
			DEBUG_SHOW(reshaped.total());
			DEBUG_SHOW(sqrt(reshaped.total()/FeatureHOG<float>::DEPTH));
			HOG_data_8U.push_back(reshaped);
		}
		HOG_data_8U.convertTo(HOG_data, CV_32F);
	} 


	if (_use_hog)
		model.svm->run(HOG_data, labels, params);
	else
		model.svm->run(data, labels, params);

	// Generate prediction scores for all training data
	Mat labels_train;
	if (_use_hog)
		model.svm->predict(HOG_data, labels_train);
	else
		model.svm->predict(data, labels_train);

	Mat labels_32F;
	labels.convertTo(labels_32F, CV_32F);
	Mat diff = labels_32F == labels_train;
	double train_true = countNonZero(diff);
	double train_pct = (train_true / (double) diff.rows) * 100.0;
	cout << "\tTraining correct: " << train_pct << "%" << endl;

	const int sv_count = model.svm->get_support_vector_count();
	const int sv_length = model.svm->get_var_count();
	cout << "\tSupport vector(s): " << sv_count << ", vector-length: " << sv_length << endl;

	CvSVMDecisionFunc* decision = model.svm->getDecisionFunc();

	const double b = decision->rho;
	cout << "line:" << __LINE__ << ") bias: " << b << endl;

	std::vector<std::pair<int, double>> sv_idx(sv_count);
	Mat sv_weight(sv_count, sv_length, CV_32F);
	Mat W = Mat::zeros(sv_length, 1, CV_32F);
	for (int i = 0; i < sv_count; ++i)
	{
		// Compute W from each support_vector and its weight: decision->alpha
		const float* support_vector = model.svm->get_support_vector(i);

		for (int j = 0; j < sv_length; ++j)
		{
			sv_weight.at<float>(i, j) = (float) -decision->alpha[i] * support_vector[j];
			W.at<float>(j) += sv_weight.at<float>(i, j);
		}

		sv_idx[i].first = i;
		sv_idx[i].second = sum(sv_weight.row(i)).val[0];
	}
	DEBUG_HERE();

	// Sort the support vectors from most pos to most neg
	std::sort(sv_idx.begin(), sv_idx.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b)
	{
		return a.second > b.second;
	});
	DEBUG_HERE();

	// Color the positive weighed support vector patches green, the neg ones red
	Mat sv_norm_weight, sv_weight_gimage, sv_weight_cimage;
	normalize(sv_weight, sv_norm_weight, 255, 0, NORM_MINMAX);
	sv_norm_weight.convertTo(sv_weight_gimage, CV_8U);
	cvtColor(sv_weight_gimage, sv_weight_cimage, COLOR_GRAY2BGR);

	DEBUG_HERE();
	const double alpha = 0.1;
	const double beta = (1.0 - alpha);
	const Mat green(1, sv_weight_cimage.cols, CV_8UC3, CV_RGB(0, 255, 0));
	const Mat red(1, sv_weight_cimage.cols, CV_8UC3, CV_RGB(255, 0, 0));
	Mat sv_weight_color_image(sv_count, sv_length, CV_8UC3);
	DEBUG_HERE();

	for(size_t r = 0; r < sv_idx.size(); ++r)
	{
		const int idx = sv_idx[r].first;

		if(sv_idx[r].second > 0) 
			addWeighted(green, alpha, sv_weight_cimage.row(idx), beta, 0.0, sv_weight_color_image.row((int) r));
		else
			addWeighted(red, alpha, sv_weight_cimage.row(idx), beta, 0.0, sv_weight_color_image.row((int) r));
	}
	DEBUG_HERE();

	// Show the support vectors
	Mat sv_tmp_sz_im_canvas;

	if (!_use_hog)
	{
		if(sv_weight.rows > 1)
		{
			const int canvas_cols = (int) sqrt((double) sv_count) + 1;
			Utility::getImagesCanvas(canvas_cols, sv_weight_color_image, _model_size, sv_tmp_sz_im_canvas, Color_BLACK);
		}
		else if(sv_weight.rows == 1)
		{
			sv_tmp_sz_im_canvas = sv_weight_cimage.reshape(sv_weight_cimage.channels(), _model_size.height);
		}


		namedWindow("Support Vector overview", CV_WINDOW_KEEPRATIO);
		imshow("Support Vector overview", sv_tmp_sz_im_canvas);
	}
	DEBUG_HERE();

	// DONE: Compute the confidence values for training and validation as the distances
	// between the sample vectors X and weight vector W, using bias b:
	// conf = X * W + b
	//  
	// Approach this as a matrix calculation (ie. fill in the dots below, using no
	// more than a single line for calculating respectively conf_train and conf_val)
	//  
	// The confidence value for training should be the same value you get from
	// svm.predict(data, labels_train);


	cout << "data size: " << data.cols << "x" << data.rows << endl;
	cout << "HOG data size: " << HOG_data.cols << "x" << HOG_data.rows << endl;
	cout << "W size: " << W.cols << "x" << W.rows << endl;
	Mat conf_train;
	if (_use_hog)
		conf_train = (HOG_data * W) + b; // data is train_data but in CV_32F
	else
		conf_train = (data * W) + b; // data is train_data but in CV_32F

	//Mat conf_val = (data * W) + b; // where is the validation data???

	Mat train_pred, train_pred_32S;
	train_pred = (conf_train > 0) / 255;
	train_pred.convertTo(train_pred_32S, CV_32S);
	train_pred_32S = (train_pred_32S * 2) - 1; //convert {0,1} to {-1,1} to match labels vector

	DEBUG_HERE();

	double train_true2 = sum((train_pred_32S == labels) / 255)[0];
	double train_pct2 = (train_true2 / (double) train_pred_32S.rows) * 100.0;

	//double val_true = sum((val_pred == val_gnd) / 255)[0];
	//double val_pct = (val_true / (double) val_pred.rows) * 100.0;

	cout << __LINE__ << "\tTraining correct: " << train_pct2 << "%" << endl;
	//cout << "\tValidation correct: " << val_pct << "%" << endl;

	model.W = W;
	model.b = b;
	model.c = C;

	DEBUG_HERE();

	//assert((int) W.total() == _model_size.area());
	//assert((int) W.total() == W_rect.total());

	// Show the model
	cout << "showing model" << endl;

	if (_use_hog)
	{
		Mat W_rect(_hog_model_size.height, _hog_model_size.width * FeatureHOG<float>::DEPTH, CV_32F);

		cout << "rectangle size:" << W_rect.cols << "x" << W_rect.rows << endl;
		cout << "data size: " << W.cols << "x" << W.rows << endl;

		W_rect.data = W.data;
		W_rect = W_rect.clone();

		//Mat weight_img = FeatureHOG<float>::visualize(W_rect);

		imshow("Model", FeatureHOG<float>::visualize(W_rect));
	} 
	else
	{
		Mat W_rect(_model_size.height, _model_size.width, CV_32F);
		W_rect.data = W.data;
		W_rect = W_rect.clone();

		normalize(W_rect, W_rect, 255, 0, NORM_MINMAX);

		Mat W_img;
		W_rect.convertTo(W_img, CV_8U);
		imshow("Model", W_img);
	}
	cout << "End of training phase!" << endl;
	//cout << "Press a key to continue" << endl;
	//waitKey();
}

/*! @brief creates a feature pyramid of several layers containing
 * 					scaled representations of the input
 *
 * @param image tphe input image
 * @param pyramid a vector of scaled features or images
 */
void Detector::createPyramid(const Mat &image, SCVMats &pyramid)
{
	Mat smallerImage = image;
	Size size;
	size.width = image.cols;
	size.height = image.rows;

	//How much to downscale the image by each pass
	float downscaleFactor = _pyramid_downscale_factor;

	//Use this factor to determine a 'stop size' in terms of the model size
	float smallestImageModelSizeFactor = _smallestImageModelSizeFactor;

	cout << "image size: (" << size.width << ", " << size.height << ")" << endl;

	//Put the original image as the bottom layer
	pyramid.push_back(std::make_shared<Mat>(image));

	//Resize the image repeatedly and add it to the pyramid as a new layer
	int stopSizeW = smallestImageModelSizeFactor * _model_size.width;
	int stopSizeH = smallestImageModelSizeFactor * _model_size.height;
	while (size.width >= stopSizeW * downscaleFactor && size.height >= stopSizeH * downscaleFactor && pyramid.size() < _pyramid_height)
	{
		size.width = smallerImage.cols / downscaleFactor;
		size.height = smallerImage.rows / downscaleFactor;
		resize(smallerImage, smallerImage, size);
		cout << "image size: (" << size.width << ", " << size.height << ")" << endl;
		pyramid.push_back(std::make_shared<Mat>(smallerImage));
	}
}

/*! @brief remove all candidates from a vector of results that
 * 					are lower than a candidate at the same location
 *
 * @param results a vector containing a struct with result locations and scores
 */
void Detector::nonMaximumSuppression(ResultLocations &results)
{
	if (!results.empty())
	{
		// Sort results by score (lambda function C++11)
		std::sort(results.begin(), results.end(), [](const ResultLocation &a, const ResultLocation &b)
		{
			return a.score > b.score;
		});

		// Start with best scoring bbox
		auto result = results.begin();
		Rect bbox(result->bbox);
		++result;

		// Vector of top bboxs (containing the best, to start with)
		Rects m_bboxs(1, bbox);

		// Iterate over the rest of the scores
		for (; result != results.end();)
		{
			bbox = result->bbox;

			double max_overlap = 0;
			for (auto m_bbox_r : m_bboxs) //forall candidate bboxs
			{
				// Calculate union over intersection of candidate bboxs en current result bbox
				const Rect intersection = m_bbox_r & bbox;
				double overlap = intersection.area() / (double) m_bbox_r.area(); // overlap in pct

				if (overlap > max_overlap)
					max_overlap = overlap;
			}

			if (max_overlap <= _nms_overlap_threshold)
			{
				// Add if max_overlap <= _nms_overlap_threshold
				m_bboxs.push_back(bbox);
				++result;
			}
			else
			{
				// Erase result otherwise (because it is worse than another result at appr. same location)
				result = results.erase(result);
			}
		}
	}
}

/*! @brief generate responses for a given image and a template model
 *
 * @param image input image
 * @param model input model
 * @param responses
 */
void Detector::getResponses(const Mat &image, const Model &model, Responses &responses)
{
	cout << "Generating model responses ..." << endl;
	assert(model.svm != nullptr);
	assert(model.svm->isTrained());
	assert(!image.empty());

	double bside = MAX(image.rows, image.cols);
	int t_height = int(image.rows * (MIN(_max_image_size, bside) / bside));
	int t_width = int(image.cols * (MIN(_max_image_size, bside) / bside));

	Mat image_sz, image_gr;
	resize(image, image_sz, Size(t_width, t_height));
	cvtColor(image_sz, image_gr, CV_BGR2GRAY);

	// Create feature pyramid
	SCVMats pyramid;
	createPyramid(image_gr, pyramid);

	cout << "Processing " << pyramid.size() << " pyramid layers" << endl;
	vector<double> fps;
	int64 t0 = Utility::get_time_curr_tick();

	for (size_t layer = 0; layer < pyramid.size(); ++layer)
	{

		/*
		 *  Slide the model template over all possible image locations
		 *  Left to Right, Top to Bottom
		 *
		 *  +------------------------------------------------------------+
		 *  |                                                            |
		 *  |                                                            |
		 *  |    +---------+ - - ->                                      |
		 *  |    |         |                                             |
		 *  |    |         |                                             |
		 *  |    |         |                                             |
		 *  |    +---------+ - - ->                                      |
		 *  |    |         |                                             |
		 *  |    v         v                                             |
		 *  |                                                            |
		 *  |                                                            |
		 *  |                                                            |
		 *  |                                                            |
		 *  |                                                            |
		 *  |                                                            |
		 *  |                                                            |
		 *  |                                                            |
		 *  +------------------------------------------------------------+
		 */

		int w, h;
		if (_use_hog && !_eval_everywhere)
		{
			w = pyramid.at(layer)->cols / FeatureHOG<float>::CELL_SIZE - _hog_model_size.width;
			h = pyramid.at(layer)->rows / FeatureHOG<float>::CELL_SIZE - _hog_model_size.height;
		}
		else
		{
			w = pyramid.at(layer)->cols - _model_size.width;
			h = pyramid.at(layer)->rows - _model_size.height;
		}
		Range rx(0, w);
		Range ry(0, h);
		Mat1i X, Y;
		Utility::meshgrid(rx, ry, X, Y);				// Generate all possible template locations
		Mat X1 = X.reshape(1, 1);
		Mat Y1 = Y.reshape(1, 1);
		if (_use_hog && !_eval_everywhere)
		{
			
			X1 *= FeatureHOG<float>::CELL_SIZE;
			Y1 *= FeatureHOG<float>::CELL_SIZE;
		}
		
		const int* px = X1.ptr<int>(0);
		const int* py = Y1.ptr<int>(0);
		vector<int> Xv(px, px + X1.total());		// Create vector of X locations
		vector<int> Yv(py, py + Y1.total());		// Create vector of Y locations
		responses.Xvs.insert(make_pair(layer, Xv));				// Map vector to to a pyramid layer X (for lookup, later)
		responses.Yvs.insert(make_pair(layer, Yv));				// Map vector to to a pyramid layer Y (for lookup, later)

		
		// Extract subwindow locations from image for detection
		Mat sub_windows((int) X1.total(), _model_size.area(), CV_32F);
		Mat mv_sub_data((int) X1.total(), _model_size.area(), CV_32F);
		Mat sv_sub_data((int) X1.total(), _model_size.area(), CV_32F);

		
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for
#endif

	if (!_use_hog || _eval_everywhere)
		for (int i = 0; i < (int) X1.total(); ++i) // for all window positions
		{
			
			Mat sub = (*pyramid.at(layer))(Rect(Point(Xv[i], Yv[i]), _model_size));
			Mat subclone;
			/*
			if (_use_hog)
			{
				cv::Mat HOG_features;
				FeatureHOG<float>::compute(sub, HOG_features);
				subclone = HOG_features.clone(); // .reshape(FeatureHOG<float>::DEPTH);
			}
			else
			*/
				subclone = sub.clone();
			Mat sub1d = subclone.reshape(1, 1);
			sub1d.convertTo(sub_windows.row(i), CV_32F);
			

			
			if (_do_equalizing)
			{
				Mat mean, stddev;
				meanStdDev(sub_windows.row(i), mean, stddev);
				sv_sub_data.row(i) = stddev.at<double>(0, 0);
				mv_sub_data.row(i) = mean.at<double>(0, 0);
			}
		}

		
		if (_do_equalizing)
		{
			// Equalize image (subtract mean, divide by stddev)
			sub_windows = sub_windows - mv_sub_data;
			sub_windows = sub_windows / sv_sub_data;
		}

		if (_do_whitening)
		{
			Utility::whiten(sub_windows, _model_size, sub_windows);
		}

		
		/*
		 * DONE: If you have found the answer to the question in the
		 * train method, you can replace the loop below by a single line.
		 *
		 * Mat detect = ...;
		 */
		cout << "detecting..."<<endl;
		Mat face_locations;
		Mat detect = Mat::zeros(sub_windows.rows, 1, CV_32F);

		// Mat detect = (sub_windows * model.W) + model.b;
		
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for
#endif

	
		if (_use_hog && !_eval_everywhere)
		{
			Mat HOG_features;
			FeatureHOG<float>::compute(*pyramid.at(layer), HOG_features);
			Mat features = HOG_features.reshape(FeatureHOG<float>::DEPTH);


			int w = features.cols - _hog_model_size.width + 1;
			int h = features.rows - _hog_model_size.height + 1;
			detect = Mat::zeros(w*h, 1, CV_32F);

			int i = 0;
			
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++)
				{
					
					Mat sub = features(Rect(Point(x, y), _hog_model_size));
					
					Mat sub1D = sub.clone().reshape(1, 1);

					//DEBUG_SHOW(_hog_model_size.width);
					//DEBUG_SHOW(_hog_model_size.height);
					//DEBUG_SHOW(_hog_model_size.height * _hog_model_size.width * FeatureHOG<float>::DEPTH);
					//DEBUG_SHOW(sub1D.total());
					//
					//
					detect.at<float>(i) = -model.svm->predict(sub1D, true); // svm->predict inverses the scores...
					i++;
				}

			
			cout << "Show detection results as a heatmap (PDF)" << endl;
			// Show detection results as a heatmap (PDF) of most likely face locations for this pyramid layer
			face_locations = (detect.reshape(detect.channels(),
				(features.size().height - _hog_model_size.height) + 1));

			
		}
		else
		{
			int hog_progress = 0;
			for (int i = 0; i < sub_windows.rows; ++i)
			{

				int hog_progress_now = i * 100 / sub_windows.rows;
				if (hog_progress_now / 2 > hog_progress / 2)
				{
					hog_progress = hog_progress_now;
					cout << hog_progress << "%" << endl;
				}

				Mat HOG_data;
				if (_use_hog)
				{
					Mat sub = (*pyramid.at(layer))(Rect(Point(Xv[i], Yv[i]), _model_size));

					//Mat row = sub_windows.row(i);
					//Mat window = row.reshape(1, _model_size.height);
					Mat window_8U = sub;
					//window.convertTo(window_8U, CV_8U);
					cv::Mat HOG_features;
					FeatureHOG<float>::compute(window_8U, HOG_features);
					Mat reshaped = HOG_features.reshape(1, 1);
					Mat HOG_32F = reshaped;
					//reshaped.convertTo(HOG_32F, CV_32F);

					detect.at<float>(i) = -model.svm->predict(HOG_32F, true); // svm->predict inverses the scores...
				}
				else
					detect.at<float>(i) = -model.svm->predict(sub_windows.row(i), true); // svm->predict inverses the scores...


			}
			cout << "Show detection results as a heatmap (PDF)" << endl;
			// Show detection results as a heatmap (PDF) of most likely face locations for this pyramid layer
			face_locations = (detect.reshape(detect.channels(),
					(pyramid.at(layer)->size().height - _model_size.height) + 1));
		}


		//Generate and show a PDF (Probabilistic Density Function)
		Mat pdf;
		Utility::getHeatmap(face_locations, pdf);
		namedWindow("Probabilistic Density Function", CV_WINDOW_KEEPRATIO);
		imshow("Probabilistic Density Function", pdf);
		waitKey(500);

		// Save range of rows for this layer
		responses.layers.push_back(Range(responses.detections.rows, responses.detections.rows + detect.rows));
		// Save responses
		responses.detections.push_back(detect);

		//Moved this to end of for loop:
		string etf = Utility::show_fancy_etf((int) layer, (int) pyramid.size(), 1, t0, fps);
		if (!etf.empty()) cout << etf << endl;
	}
}

/*! @brief generate a precision and recall score by comparing each detection to the ground truth
 *
 * @param ground_truths vector of ground truth rectangles
 * @param results a vector containing a struct with result locations and scores
 * @return pair of precision and recall percentages (0 to 1)
 */
std::pair<double, double> Detector::precisionRecall(const Rects &ground_truths,
		const ResultLocations &results)
{
	std::pair<double, double> precision_recall(0, 0);

	if (!ground_truths.empty())
	{
		vector<Rect> search_gt = ground_truths;
		int tp = 0; //True positives counter
		for (size_t r = 0; r < results.size(); ++r)
		{
			Rect detection = results.at(r).bbox;

			vector<Rect>::iterator it = search_gt.begin();
			for (; it != search_gt.end();)
			{
				Rect ground_truth = *it;
				/*
				* Calculate the intersection size as the intersection-over-union between
				* the detection result and the ground_truth
				*/
				int insct = (ground_truth & detection).area(); //Intersection area
				double intersection_size = (insct + 0.0) / (ground_truth.area() + detection.area() - insct); //Intersection over union

				if (intersection_size >= _gt_accuracy)
				{
					// if size is equal or larger than the accuracy threshold we have found a true positive
					it = search_gt.erase(it);
					tp++;
				}
				else
				{
					++it;
				}
			}
		}

		// Calculate the Precision and Recall at the currect threshold value
		int rs = results.size(), ts = ground_truths.size();
		double precision = 0, recall = 0;
		if (rs > 0) {
			precision = (tp + 0.0) / rs; //yay integer division
		}
		if (ts > 0) {
			recall = (tp + 0.0) / ts;
		}

		cout << precision << "," << recall << " (Threshold,Precision,Recall)" << endl;
		precision_recall.first = precision;
		precision_recall.second = recall;
	}

	return precision_recall;
}

/*! @brief draw the detection results onto a canvas containing the input image
 *
 * @param canvas the painting canvas
 * @param image the input image
 * @param results a vector containing a struct with result locations and scores
 * @param max_value threshold slider max value
 */
void Detector::drawResults(Mat &canvas, const ResultLocations &results)
{
	// Convert image to 3 channels, just so we can place bounding boxes in red
	if (canvas.channels() == 1)
		cvtColor(canvas, canvas, CV_GRAY2BGR);

	for (size_t r = 0; r < results.size(); ++r)
	{
		Rect rect = results.at(r).bbox;
		rectangle(canvas, rect, Color_RED);

		stringstream text;
		text << results.at(r).score;
		double scale = 0.8;
		int font = CV_FONT_HERSHEY_PLAIN;
		int thickness = 1;
		int baseline = 0;
		Size size = getTextSize(text.str(), font, scale, thickness, &baseline);
		rectangle(canvas, rect, Color_RED, 2);
		putText(canvas, text.str(), Point(rect.x, rect.y + size.height), font, scale, Color_BLACK, thickness + 1, CV_AA);
		putText(canvas, text.str(), Point(rect.x, rect.y + size.height), font, scale, Color_YELLOW, thickness, CV_AA);
	}
}

/*! @brief main program runner
 */
void Detector::run()
{
	assert(FileIO::isFile(_query_image_file));

	////////////////////// Read pos and neg examples ////////////////////////////
	vector<string> pos_files, neg_files;
	readPosFilelist(pos_files);
	readNegFilelist(neg_files);

	assert((int) pos_files.size() > _pos_amount);
	assert((int) neg_files.size() > 0);
	int neg_amount = MIN((int) neg_files.size() / 2, _pos_amount * _posneg_factor / 2);

	cout << "Total images, pos:" << pos_files.size() << ", neg:" << neg_files.size() << endl;

	vector<string> pos_train(pos_files.begin(), pos_files.begin() + _pos_amount);
	vector<string> pos_val(pos_files.begin() + _pos_amount,
			pos_files.begin() + MIN(2 * _pos_amount, (int) pos_files.size())); //Fixed bug
	vector<string> neg_train(neg_files.begin(), neg_files.begin() + neg_amount);
	vector<string> neg_val(neg_files.begin() + neg_amount,
			neg_files.begin() + MIN(2 * neg_amount, (int) neg_files.size())); //Fixed bug

	cout << "Positive images used:" << pos_train.size() << ", validation:" << pos_val.size() << endl;
	cout << "Negative images used:" << neg_train.size() << ", validation:" << neg_val.size() << endl; 

	Mat pos_tmp_im = imread(pos_files.front(), CV_LOAD_IMAGE_GRAYSCALE);
	int width = _target_width;
	int height = int(width * double(pos_tmp_im.rows / (double) pos_tmp_im.cols));
	const Size sz_dim = Size(width, height);
	Mat pos_tmp_sz_im;
	resize(pos_tmp_im, pos_tmp_sz_im, sz_dim);

	Mat pos_train_data, neg_train_data;
	cout << endl << "line:" << __LINE__ << ") Read training images" << endl;
	cout << "==============================" << endl;
	readPosData(pos_train, pos_train_data);
	readNegData(neg_train, neg_train_data);
	/////////////////////////////////////////////////////////////////////////////

	cout << "model size: " << _model_size.width << "x" << _model_size.height << endl;
	_hog_model_size = Size(_model_size.width / FeatureHOG<float>::CELL_SIZE + FeatureHOG<float>::CELL_SIZE % 2, _model_size.height / FeatureHOG<float>::CELL_SIZE + FeatureHOG<float>::CELL_SIZE % 2);
	cout << "hog model size: " << _hog_model_size.width << "x" << _hog_model_size.height << endl;

	cout << "training images have been read." << endl;

	/////////////////////////// Whitening transformation ////////////////////////
	Mat whitened_pos_data, whitened_neg_data;
	if (_do_whitening)
	{
		Utility::whiten(pos_train_data, _model_size, whitened_pos_data);
		Utility::whiten(neg_train_data, _model_size, whitened_neg_data);
	}
	/////////////////////////////////////////////////////////////////////////////
	cout << "whitened" << endl;

	////////////////////// Show pos and neg examples ////////////////////////////

	const int canvas_total = MIN(_disp, MIN(neg_train_data.rows, pos_train_data.rows));
	const int canvas_cols = (int) sqrt((double) canvas_total);

	Mat pos_tmp_sz_im_canvas, neg_tmp_sz_im_canvas;
	Utility::getImagesCanvas(canvas_cols, pos_train_data, sz_dim, pos_tmp_sz_im_canvas);
	Utility::getImagesCanvas(canvas_cols, neg_train_data, sz_dim, neg_tmp_sz_im_canvas);


	if (_do_whitening)
	{
		Mat whitened_pos_tmp_sz_im_canvas;
		Utility::getImagesCanvas(canvas_cols, whitened_pos_data, sz_dim, whitened_pos_tmp_sz_im_canvas);
		Mat line = Mat::zeros(Size(2, pos_tmp_sz_im_canvas.rows), pos_tmp_sz_im_canvas.type());
		hconcat(pos_tmp_sz_im_canvas, line, pos_tmp_sz_im_canvas);
		hconcat(pos_tmp_sz_im_canvas, whitened_pos_tmp_sz_im_canvas, pos_tmp_sz_im_canvas);
	}
	/////////////////////////////////////////////////////////////////////////////

	//////////// Read ground truth if we consider data/img1.jpg /////////////////
	vector<Rect> ground_truths;
	if (_query_image_file == Detector::cfg()->getValue<string>("settings/images/test/ground_truth@file"))
	{
		QXmlElms faces_xml = Detector::cfg()->getValues("settings/images/test/ground_truth/face");
		for (size_t r = 0; r < faces_xml.size(); ++r)
		{
			QXmlElmPtr face_xml = faces_xml[r];
			int id = Detector::cfg()->getValue<int>("@id", face_xml);
			int width = Detector::cfg()->getValue<int>("width", face_xml);
			int height = Detector::cfg()->getValue<int>("height", face_xml);
			int x = Detector::cfg()->getValue<int>("x", face_xml);
			int y = Detector::cfg()->getValue<int>("y", face_xml);
			ground_truths.push_back(Rect(x, y, width, height));
		}
	}
	/////////////////////////////////////////////////////////////////////////////

	///////////// Build train / val datasets and display stuff  /////////////////
	cout << endl << "line:" << __LINE__ << ") Read validation images" << endl;
	cout << "==============================" << endl;
	Mat pos_val_data, neg_val_data;
	readPosData(pos_val, pos_val_data);
	readNegData(neg_val, neg_val_data);

	Mat whitened_pos_val_data, whitened_neg_val_data;
	if (_do_whitening)
	{
		Utility::whiten(pos_val_data, _model_size, whitened_pos_val_data);
		Utility::whiten(neg_val_data, _model_size, whitened_neg_val_data);
	}

	Mat val_data;
	val_data.push_back(_do_whitening ? whitened_pos_val_data : pos_val_data);
	val_data.push_back(_do_whitening ? whitened_neg_val_data : neg_val_data);

	cout << "Show windows" << endl;
	namedWindow("Pos examples", CV_WINDOW_KEEPRATIO);
	namedWindow("Neg examples", CV_WINDOW_KEEPRATIO);
	namedWindow("Pos mean image", CV_WINDOW_KEEPRATIO);
	namedWindow("Neg mean image", CV_WINDOW_KEEPRATIO);
	imshow("Pos examples", pos_tmp_sz_im_canvas);
	imshow("Neg examples", neg_tmp_sz_im_canvas);
	imshow("Pos mean image", _pos_sum8U);
	imshow("Neg mean image", _neg_sum8U);
	//cout << "Press a key to continue" << endl;
	//waitKey();
	namedWindow("Model", CV_WINDOW_KEEPRATIO);

	Mat val_labels(pos_val_data.rows, 1, CV_32S, Scalar::all(1));
	val_labels.push_back(Mat(neg_val_data.rows, 1, CV_32S, Scalar::all(-1)));
	Mat val_gnd = ((val_labels > 0) / 255) * 2 - 1;
	/////////////////////////////////////////////////////////////////////////////

	//////////////////// Test model from mean of images /////////////////////////
	Mat alt_pred = (val_data * _pos_sumF.t() > 0) / 255;
	double alt_true = alt_pred.size().height - sum((alt_pred == val_gnd) / 255)[0];
	double alt_pct = (alt_true / (double) alt_pred.size().height) * 100.0;
	cout << "Validation correct with mean model: " << alt_pct << "%" << endl;
	/////////////////////////////////////////////////////////////////////////////

	/////////////////////////////// Train SVM ///////////////////////////////////
	Model model;														// initialize a new model

	Mat train_data;
	train_data.push_back(_do_whitening ? whitened_pos_data : pos_train_data);
	train_data.push_back(_do_whitening ? whitened_neg_data : neg_train_data);

	Mat pos_labels = Mat(pos_train_data.rows, 1, CV_32S, Scalar::all(1));
	Mat neg_labels = Mat(neg_train_data.rows, 1, CV_32S, Scalar::all(-1));

	Mat train_labels;
	train_labels.push_back(pos_labels);
	train_labels.push_back(neg_labels);

	train(train_data, train_labels, model);		// train it based on pos/neg train data
	/////////////////////////////////////////////////////////////////////////////
	cout << "training finished. start testing" << endl;
	////////////////////////////// Test on real image ///////////////////////////
	const Mat image = imread(_query_image_file);

	// Book keeping of image resizing factor for bounding box
	const double bside = MAX(image.rows, image.cols);
	const double factor = 1 / (MIN(_max_image_size, bside) / bside);
	const Size im_size(int(image.cols / factor), int(image.rows / factor));
	
	Responses responses;
	getResponses(image, model, responses);


	double min_val, max_val;
	minMaxLoc(responses.detections, &min_val, &max_val);
	std::cout << "line:" << __LINE__ << ") response range: " << min_val << " <-> " << max_val << std::endl;

	// Create window
	cout << "creatiung windows" << endl;
	namedWindow("Search image", CV_WINDOW_KEEPRATIO);
	resizeWindow("Search image", im_size.width, im_size.height);

	// Trackbar with default values for detection threshold
	int value = _initial_threshold - _min_slider_value, o_val = -INT_MAX;
	createTrackbar("Threshold", "Search image", &value, _max_slider_value - _min_slider_value);


	cout << "creatiung Detection results" << endl;
	// Detection results
	ResultLocations results;

	std::string prec_rec_line = "";

	const Size box_size(int(_model_size.width * factor), int(_model_size.height * factor));

	// Drawing loop (to be able to vary the threshold
	int key = -1;
	while (true)
	{
		int o_value = _min_slider_value + value;
		if (o_value != o_val)
		{
			results.clear();		// Generate new results

			double threshold = ((_max_slider_value - o_value) / 100.0) * _max_slider_value;

			cout << (50 - threshold) << ",";

			// Create result vector with all detections above the threshold
			for (int i = 0; i < responses.detections.rows; ++i)
			{
				const double r_pct = _max_slider_value - 
						(((responses.detections.at<float>(i) - min_val) /
								(max_val - min_val)) * _max_slider_value);

				if (r_pct < threshold)
				{
					// Find correct image location for this detection considering the pyramid layer
					int offset = 0;
					int layer_n = 0;
					for (size_t l = 0; l < responses.layers.size(); ++l)
					{
						Range layer = responses.layers.at(l);

						if (i >= layer.start && i < layer.end)
						{
							layer_n = (int) l;
							break;
						}
						offset += layer.size();
					}

					ResultLocation result;
					result.score = responses.detections.at<float>(i);

					const Point point(int(responses.Xvs[layer_n][i - offset] * factor), int(responses.Yvs[layer_n][i - offset] * factor));
					result.bbox = Rect(point, box_size);
					results.push_back(result);
				}
			}

			nonMaximumSuppression(results);

			auto precision_recall = precisionRecall(ground_truths, results);

			char p_buf[32], r_buf[32];
#ifdef __linux__
			sprintf(p_buf, "%4.2f", precision_recall.first);
			sprintf(r_buf, "%4.2f", precision_recall.second);
#else
			sprintf_s(p_buf, "%4.2f", precision_recall.first);
			sprintf_s(r_buf, "%4.2f", precision_recall.second);
#endif

			std::stringstream prss;
			prss << "Precision: " << p_buf << "% Recall: " << r_buf << "%";
			prec_rec_line = prss.str();

			o_val = o_value;
		}

		Mat canvas = image.clone();

		if(_show_ground_truth)
			for(auto ground_truth : ground_truths)
				rectangle(canvas,	ground_truth, Color_GREEN, 2, CV_AA);
		drawResults(canvas, results);
		putText(canvas, prec_rec_line, Point(4, 10), CV_FONT_HERSHEY_PLAIN, .6, Color_BLACK, 2, CV_AA);
		putText(canvas, prec_rec_line, Point(4, 10), CV_FONT_HERSHEY_PLAIN, .6, Color_WHITE, 1, CV_AA);

		imshow("Search image", canvas);
		key = waitKey(50);

		if (key == 'g')         // Show/hide ground truth bounding boxes
			_show_ground_truth = !_show_ground_truth;
		else if(key == 2424832) // Left key (threshold slider lower)
			value--;
		else if(key == 2555904) // Right key (threshold slider higher)
			value++;
		else if(key == 27)      // Quit
			break;

		if(value < 0) value = 0;
		if(value > _max_slider_value - _min_slider_value) value = _max_slider_value - _min_slider_value;
		setTrackbarPos("Threshold", "Search image", value);
	}
	/////////////////////////////////////////////////////////////////////////////
}

} /* namespace nl_uu_science_gmt */
