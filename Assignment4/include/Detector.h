/*
 * Detector.h
 *
 *  Created on: Aug 23, 2013
 *      Author: coert
 */

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <opencv2/core/core.hpp>
#include <QueryXML.h>
#include <stddef.h>
#include <cfloat>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace nl_uu_science_gmt
{
#ifdef DEBUG
const static int NUM_THREADS = 1;
#else
const static int NUM_THREADS = cv::getNumberOfCPUs();
#endif

class MySVM;

class Detector
{
	static QueryXML* _Config;

	/**
	 * Template model (the SVM hyperplane)
	 */
	struct Model
	{
		std::shared_ptr<MySVM> svm;		// OpenCV support vector machine

		cv::Mat W;				// weight vector (1D)
		double b;					// bias
		double c;					// SVM fault tolerance

		Model()
		{
			svm = nullptr;
			b = -DBL_MAX;
			c = -DBL_MAX;
		}
	};

	/**
	 * Connection between response matrix scores and pyramid layers & layer locations
	 */
	struct Responses
	{
		// Bounding box locations per pyramid layer
		std::map<int, std::vector<int>> Xvs, Yvs;

		// Every pyramid layer's responses are represented by
		//   a range of rows in the detection response matrix
		std::vector<cv::Range> layers;

		// Detection Response Matrix
		// All response scores at every detection location of every pyramid layer
		cv::Mat detections;
	};

	/**
	 * Bounding box locations with response scores
	 */
	struct ResultLocation
	{
		double score;			// Template score
		cv::Rect bbox;		// Template location

		ResultLocation()
		{
			score = -DBL_MAX;
		}
	};

	typedef std::vector<std::shared_ptr<cv::Mat>> SCVMats;
	typedef std::vector<ResultLocation> ResultLocations;
	typedef std::vector<std::string> Strings;
	typedef std::vector<cv::Rect> Rects;

	const int _seed;                        // Randomizing seed

	const int _use_hog;                  // (TK) whether to use HOG features instead of raw pixel data

	const int _pos_amount;                  // Amount of positive learning data (images)
	const int _target_width;                // Target image width (height will be relative the width)
	const int _posneg_factor;               // 4x as many negative images as positive ones
	const bool _do_equalizing;              // Equalize images by subtracting mean and dividing by stddev
	const bool _do_whitening;               // Equalize images by matrix whitening
	bool _show_ground_truth;                // Show ground truth bounding boxes (if applicable)

	const int _disp;												// Desired image amount for training image overview canvas
	const std::string _query_image_file;		// Input qurey image to perform detection on after training
	const int _max_image_size;							// Max query image width or height (larger ones will be resized to this)

	const size_t _max_count;								// SVM the maximum number of iterations or elements to compute
	const double _epsilon;									// SVM the desired accuracy or change in parameters at which the iterative algorithm stops

	const double _nms_overlap_threshold;		// Non Maxima Suppression threshold (as pct. of union over intersection area)

	const int _initial_threshold;						// Initial detection threshold value
	const int _min_slider_value;						// Detection threshold slider min value
	const int _max_slider_value;						// Detection threshold slider max value

	const double _gt_accuracy;							// Accuracy threshold for determining true positive ground truth overlap

	cv::Size _model_size;										// Template model Width x Height

	cv::Mat _neg_sum8U, _pos_sum8U, _pos_sumF; // Mean image representations (pos image, neg image and pos float)

	void readPosFilelist(Strings &);
	void readNegFilelist(Strings &);

	void readPosData(const Strings &, cv::Mat &);
	void readNegData(const Strings &, cv::Mat &);

	void train(const cv::Mat &, const cv::Mat &, Model &);
	void getResponses(const cv::Mat &, const Model &, Responses &);

	void createPyramid(const cv::Mat &, SCVMats &);
	void nonMaximumSuppression(ResultLocations &);
	std::pair<double, double> precisionRecall(const Rects &, const ResultLocations &);

	static inline void drawResults(cv::Mat &, const ResultLocations &);

public:
	Detector(const std::string &);
	virtual ~Detector();

	void run();

	static std::string StorageExt;
	static std::string ImageExt;

	static QueryXML* cfg();
	static void cfg_destroy();
};

} /* namespace nl_uu_science_gmt */
#endif /* DETECTOR_H_ */
