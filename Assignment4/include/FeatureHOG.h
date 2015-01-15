/*
 * FeatureHOG.h
 *
 *  Created on: Feb 15, 2013
 *      Author: coert
 */

#ifndef FEATUREHOG_H_
#define FEATUREHOG_H_

#include <opencv2/core/core.hpp>

#ifdef __linux__
#include <cstdbool>
#endif
#include <vector>

#include <opencv2/opencv.hpp>

namespace nl_uu_science_gmt
{

template<typename T>
class FeatureHOG
{
public:
	//! minimal value (avoid division by 0)
	const static T EPS;
	//! pixel width & height inside a HOG cell
	const static int CELL_SIZE;
	//! amount of bins in a HOG cell
	const static int BINS;
	//! matrix depth (slices)
	const static int DEPTH;
	//! amount of interpolation margin (4)
	const static int TEXTURES;
	//! histogram margin
	const static int MARGIN;
	//! truncation value
	const static T TRUNC;
	//! texture energy features offset value (set empirically?)
	const static T TEXTURE;
	//! magnitude offset value (0)
	const static T STEP;
	//! use texture energy features?
	const static bool USE_TEXTURE_FEATURES;
	//! use occlusion feature (see code)?
	const static bool USE_OCCLUSION_FEATURE;

private:
	enum Surroundings
	{
		SurN, SurS, SurE, SurW
	};

	enum Type
	{
		HOG_VALS
	};

	struct Gradient
	{
		T dx;
		T dy;
		T v;
	};

	template<typename TI>
	static inline const TI square(TI val)
	{
		return val * val;
	}

	const static std::vector<T>& LUT_COS;
	const static std::vector<T>& LUT_SIN;
	const static std::vector<int>& LUT_FLIPPED_PERMUTATIONS;

	static inline const std::vector<T> calcCosLUT();
	static inline const std::vector<T> calcSinLUT();
	static inline cv::Mat rotateImage(const cv::Mat &, double);
	static inline const cv::Mat fold(const cv::Mat &);

	static inline std::vector<int> calcPermutations();
	template<typename IT>
	static inline void getSurroundingPixels(const cv::Point &p, const cv::Mat_<IT> &image, std::vector<IT> &);
	template<typename U>
	static inline void getMagnitudeMono(const std::vector<U> &, Gradient &);
	template<typename U>
	static inline void getMagnitudeColor(const std::vector<U> &, Gradient &);
	static inline void getGradientData(const cv::Point&, const cv::Mat &, Gradient &);
	static const cv::Mat createVisualization(const cv::Mat &, const int = 32);

public:
	static void compute(const cv::Mat &, cv::Mat &, int = 0);
	static void flipFeatures(cv::Mat &);
	static const cv::Mat visualize(const std::vector<T> &, const cv::Size &, int = 32);
	static const cv::Mat visualize(const cv::Mat &, const int = 32);
};

}
/* namespace nl_uu_science_gmt */
#endif /* FEATUREHOG_H_ */
