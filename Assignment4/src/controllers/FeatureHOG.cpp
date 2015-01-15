/*
 * FeatureHOG.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: coert
 */

#include <FeatureHOG.h>

#include <Detector.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <QueryXML.h>
#include <stddef.h>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

namespace nl_uu_science_gmt
{

template<typename T> const T FeatureHOG<T>::EPS = (T) 1e-6;
template<typename T> const int FeatureHOG<T>::CELL_SIZE = Detector::cfg()->getValue<int>("settings/features/hog/cell_size");
template<typename T> const int FeatureHOG<T>::BINS = Detector::cfg()->getValue<int>("settings/features/hog/gradient_bins");
template<typename T> const int FeatureHOG<T>::MARGIN = Detector::cfg()->getValue<int>("settings/features/hog/margin");
template<typename T> const T FeatureHOG<T>::TRUNC = Detector::cfg()->getValue<T>("settings/features/hog/truncation");
template<typename T> const T FeatureHOG<T>::TEXTURE = Detector::cfg()->getValue<T>("settings/features/hog/texture");
template<typename T> const T FeatureHOG<T>::STEP = Detector::cfg()->getValue<T>("settings/features/hog/step");
template<typename T> const bool FeatureHOG<T>::USE_TEXTURE_FEATURES = Detector::cfg()->getValue<bool>("settings/features/hog/use_texture_features");
template<typename T> const bool FeatureHOG<T>::USE_OCCLUSION_FEATURE = Detector::cfg()->getValue<bool>("settings/features/hog/use_occlusion_features");
template<typename T> const int FeatureHOG<T>::TEXTURES = square(Detector::cfg()->getValue<int>("settings/features/hog/margin") + 1);
template<typename T> const int FeatureHOG<T>::DEPTH = 3 * Detector::cfg()->getValue<int>("settings/features/hog/gradient_bins") + 
	(Detector::cfg()->getValue<bool>("settings/features/hog/use_texture_features") ? square(Detector::cfg()->getValue<int>("settings/features/hog/margin") + 1) : 0) + 
	(Detector::cfg()->getValue<bool>("settings/features/hog/use_occlusion_features") ? 1 : 0); // Win32 compatibility :-/
template<typename T> const std::vector<T>& FeatureHOG<T>::LUT_COS = FeatureHOG<T>::calcCosLUT();
template<typename T> const std::vector<T>& FeatureHOG<T>::LUT_SIN = FeatureHOG<T>::calcSinLUT();
template<typename T> const std::vector<int>& FeatureHOG<T>::LUT_FLIPPED_PERMUTATIONS = FeatureHOG<T>::calcPermutations();

// *************************************** Felzenszwalb HOG ***********************************************

/**
 * Non-member function to statically compute HOG descriptor from an image the 'Felzenszwalb' way
 *
 * Example: FeatureHOG<float>::compute(image, features);
 *
 * cv::Mat features;
 * An interleaved matrix containing the HOG cell data which consists of cells with bins. The histogram bins
 * are interleaved with the cell columns:
 * Mat.height: "the vertical amount of cells"
 * Mat.width:  "the horizontal amount of cells x FeatureHOG<T>::DEPTH"
 * This interleaves DEPTH slices  (= amount of bins per cell) into the X columns,
 * so HOG_cell_width = Mat.width / FeatureHOG<T>::DEPTH
 *
 * Ie.:
 * Y0X0S0, Y0X0S1, Y0X0S2, Y0X0S3, ..., Y0X0S31, Y0X1S0, ..., Y0X19S1, Y0X19S2, Y0X19S3, ..., Y0X19S31
 * Y1X0S0, Y1X0S1, Y1X0S2, Y1X0S3, ..., Y1X0S31, Y1X1S0, ..., Y1X19S1, Y1X19S2, Y1X19S3, ..., Y1X19S31
 * Y2X0S0, Y2X0S1, Y2X0S2, Y2X0S3, ..., Y2X0S31, Y2X1S0, ..., Y2X19S1, Y2X19S2, Y2X19S3, ..., Y2X19S31
 * .
 * .
 *
 * FeatureHOG<T>::DEPTH = The depth of the matrix, with each slice containing the:
 * contrast sensitive, contrast insensitive, texture energy and occlusion feature bins
 *
 * Use cv::split and cv::merge to get access to each slice for convolution. Eg.:
 *
 * cv::Mat features;
 * FeatureHOG<float>::compute(image, features);
 * std::vector<cv::Mat> feature_slices;
 * cv::split(features.reshape(FeatureHOG<T>::DEPTH), feature_slices); // Reshape to DEPTH channels, and split into vector of channel slices
 * --operate on feature slices--
 * cv::merge(feature_slices, features);                               // Merge into single matrix
 * features.reshape(1).copyTo(features);                              // Re-interleave all channel slices
 *
 * @return
 */
template<typename T>
void FeatureHOG<T>::compute(const cv::Mat &image, cv::Mat &features, int cell_size)
{
	if (cell_size == 0) cell_size = CELL_SIZE;
	assert(image.type() == CV_8U || image.type() == CV_8UC3);

	cv::Size blocks;
	blocks.height = (int) cvRound(image.rows / (T) cell_size);
	blocks.width = (int) cvRound(image.cols / (T) cell_size);

	const cv::Size visible(blocks.width * cell_size, blocks.height * cell_size);

	if (blocks.height > 0 && blocks.width > 0)
	{
		cv::Mat padded_image;

		/*
		 * Pad the image such that its width and height are an exact multiple of cell_size
		 */
		const int cm = cell_size * MARGIN;
		const int border_h = image.rows - visible.height;
		const int border_v = image.cols - visible.width;
		if (border_h != 0 || border_v != 0)
		{
			const int top = (int) (border_h / (T) 2.0);
			const int bottom = border_h - top;
			const int left = (int) (border_v / (T) 2.0);
			const int right = border_v - left;

			cv::copyMakeBorder(image, padded_image, -top + cm, -bottom + cm, -left + cm, -right + cm, cv::BORDER_REPLICATE);
		}
		else
		{
			cv::copyMakeBorder(image, padded_image, cm, cm, cm, cm, cv::BORDER_REPLICATE);
		}
		assert(padded_image.rows % cell_size == 0 && padded_image.cols % cell_size == 0);

		// memory for HOG features dimensions
		const int min_size = 2 * MARGIN;
		const int matrix_dims[] = { DEPTH, padded_image.rows / cell_size, padded_image.cols / cell_size };
		const int dims[] = { BINS * 2, matrix_dims[1], matrix_dims[2] };
		const int hog_dims[] = { DEPTH, matrix_dims[1] - min_size, matrix_dims[2] - min_size };

		// gradient histogram
		cv::Mat hist(dims[1], dims[2] * dims[0], cv::DataType<T>::type, cv::Scalar::all(0));
		const size_t hist_s = hist.step1();
		T* const hist_p = hist.ptr<T>(0);
		const int hist_width = hist.cols / dims[0];

		/*
		 * Start at x=1,y=1 because we look around the pixel under consideration
		 */
		for (int iy = 1; iy < padded_image.rows - 1; ++iy)
		{
			for (int ix = 1; ix < padded_image.cols - 1; ++ix)
			{
				// add to 4 histograms around pixel using linear interpolation
				const T xp = ((T) ix + (T) 0.5) / (T) cell_size - (T) 0.5;
				const T yp = ((T) iy + (T) 0.5) / (T) cell_size - (T) 0.5;
				const int ixp = (int) floor(xp);
				const int iyp = (int) floor(yp);

				Gradient gradient;
				getGradientData(cv::Point(ix, iy), padded_image, gradient);

				// snap to one of 18 orientations
				T best_dot = 0;
				int best_o = 0;
				for (int o = 0; o < BINS; o++)
				{
					const T dot = LUT_COS[o] * gradient.dx + LUT_SIN[o] * gradient.dy;

					if (dot > best_dot)
					{
						best_dot = dot;
						best_o = o;
					}
					else if (-dot > best_dot)
					{
						best_dot = -dot;
						best_o = o + BINS;
					}
				}

				const T vx0 = xp - ixp;
				const T vy0 = yp - iyp;
				const T vx1 = (T) 1.0 - vx0;
				const T vy1 = (T) 1.0 - vy0;

				// only add to histogram if within range
				if (iyp >= 0 && ixp >= 0 && iyp < hist.rows && ixp < hist_width)
				{
					T* dst = hist_p + iyp * hist_s + ixp * dims[0] + best_o;
					*dst += vx1 * vy1 * gradient.v;
				}
				if (iyp >= 0 && ixp + 1 >= 0 && iyp < hist.rows && ixp + 1 < hist_width)
				{
					T* dst = hist_p + iyp * hist_s + (ixp + 1) * dims[0] + best_o;
					*dst += vx0 * vy1 * gradient.v;
				}
				if (iyp + 1 >= 0 && ixp >= 0 && iyp + 1 < hist.rows && ixp < hist_width)
				{
					T* dst = hist_p + (iyp + 1) * hist_s + ixp * dims[0] + best_o;
					*dst += vx1 * vy0 * gradient.v;
				}
				if (iyp + 1 >= 0 && ixp + 1 >= 0 && iyp + 1 < hist.rows && ixp + 1 < hist_width)
				{
					T* dst = hist_p + (iyp + 1) * hist_s + (ixp + 1) * dims[0] + best_o;
					*dst += vx0 * vy0 * gradient.v;
				}
			}
		}

		// normalization matrix
		cv::Mat norm(dims[1], dims[2], cv::DataType<T>::type, cv::Scalar::all(0));
		const size_t norm_s = norm.step1();
		T* const norm_p = norm.ptr<T>(0);

		// compute energy in each block by summing over all orientations
		for (int y = 0; y < dims[1]; ++y)
		{
			const T* src = hist_p + y * hist_s;
			assert(*src == *src);
			T* dst = norm_p + y * norm_s;
			T const * const dst_end = dst + dims[2];

			while (dst < dst_end)
			{
				assert(*dst == 0);

				for (int o = 0; o < BINS; ++o)
				{
					*dst += square(*src + *(src + BINS));
					assert(*dst == *dst);
					assert(*dst > -INT_MAX && *dst < INT_MAX);
					src++;
				}

				src += BINS;
				dst++;
			}
		}

		// feature matrix
		features = cv::Mat(hog_dims[1], hog_dims[2] * hog_dims[0], cv::DataType<T>::type);
		const size_t feature_s = features.step1();
		T* const features_p = features.ptr<T>(0);

		for (int x = 0; x < hog_dims[2]; ++x)
		{
			for (int y = 0; y < hog_dims[1]; ++y)
			{
				const int mx = x + 1;
				const int my = y + 1;

				/*
				 * (MARGIN + 1)^2 interpolation areas around (x,y)
				 *
				 * with MARGIN=1, interpolation over 4x4 cells:
				 *
				 * +---+---+---+
				 * |   |   |   |
				 * +---+---+---+
				 * |   |n0 |n1 |
				 * +---+---+---+
				 * |   |n2 |n3 |
				 * +---+---+---+
				 *
				 * +-----+-----+
				 * | p0  | p1  |
				 * |     |     |
				 * +-----+-----+
				 * | p2  | p3  |
				 * |     | n.  |
				 * +-----+-----+
				 */
				std::vector<T> n;
				for (int ny = 0; ny < MARGIN + 1; ++ny)
				{
					for (int nx = 0; nx < MARGIN + 1; ++nx)
					{
						std::vector<T> p;
						for (int py = -MARGIN; py < MARGIN; ++py)
							for (int px = -MARGIN; px < MARGIN; ++px)
							{
								assert((my + ny + py) < dims[1] && (mx + nx + px) < dims[2]);
								T val = *(norm_p + (my + ny + py) * norm_s + (mx + nx + px));
								assert(val == val);
								assert(val > -INT_MAX && val < INT_MAX);
								p.push_back(val);
							}

						T val = (T) 1.0 / sqrt(std::accumulate(p.begin(), p.end(), EPS));
						assert(val == val);
						assert(val > -INT_MAX && val < INT_MAX);
						n.push_back(val);
					}
				}

				const T* src = hist_p + my * hist_s + mx * dims[0];
				T* dst = features_p + y * feature_s + x * DEPTH;

				/*
				 * contrast-sensitive features (opposing bins have not same magnitude)
				 */
				const T* src_s = src;
				std::vector<T> t(n.size(), 0);
				for (int o = 0; o < 2 * BINS; o++)
				{
					std::vector<T> h(n.size());
					for (size_t i = 0; i < n.size(); ++i)
						h[i] = std::min(*src_s * n[i], TRUNC);

					if (USE_TEXTURE_FEATURES)
					{
						t[0] += h[0];
						t[1] += h[1];
						t[2] += h[3];
						t[3] += h[2];
						assert(t[0] == t[0] && t[1] == t[1] && t[2] == t[2] && t[3] == t[3]);
					}

					*dst = (T) 0.5 * std::accumulate(h.begin(), h.end(), (T) 0.0) - STEP;
					assert(*dst == *dst);
					++dst;
					++src_s;
				}

				/*
				 * contrast-insensitive features (opposing bins have same magnitude)
				 */
				src_s = src;
				for (int o = 0; o < BINS; o++)
				{
					T sum = *src_s + *(src_s + BINS);

					std::vector<T> h(n.size());
					for (size_t i = 0; i < n.size(); ++i)
						h[i] = std::min(sum * n[i], TRUNC);

					*dst = (T) 0.5 * std::accumulate(h.begin(), h.end(), (T) 0.0) - STEP;
					assert(*dst == *dst);
					++dst;
					++src_s;
				}

				/*
				 * Texture features (general pixel intensity)
				 */
				if (USE_TEXTURE_FEATURES)
				{
					for (int i = 0; i < (int) t.size(); ++i)
					{
						*dst = TEXTURE * t[i] - STEP;
						assert(*dst == *dst);
						++dst;
					}
				}

				/*
				 * Occlusion feature
				 *
				 * Please see section 2.3 in http://cs.brown.edu/~pff/latent-release4/release4-notes.pdf
				 * Ross
				 *
				 * The boundary occlusion feature enables the learning of a bias parameter for each filter
				 * cell that is added to the filter response if that filter cell is placed in the boundary region
				 */
				if (USE_OCCLUSION_FEATURE)
				{
					*(dst++) = 0;
				}
			}
		}
	}

	// fails if there's NAN in the matrix (DEBUG)
	assert((cv::sum(features) == cv::sum(features)));
}

/**
 * Generate normalized visualization of a given HOG features matrix with given pixels per cell (cell_size)
 *
 * Example:
 * cv::Mat features;
 * FeatureHOG<float>::compute(image, features);
 * cv::Mat visualization = FeatureHOG<float>::visualize(features);
 * cv::imshow("HOG", visualization);
 * cv::waitKey();
 *
 * @return
 */
template<typename T>
const cv::Mat FeatureHOG<T>::visualize(const cv::Mat &features, int cell_size)
{
	cv::Mat ftrs = features;
	if(cv::DataType<T>::type != features.type())
		features.convertTo(ftrs, cv::DataType<T>::type);

	cv::Mat w_pos = fold(ftrs);  // DEPTH slices -> BINS slices
	cv::Mat w_neg = cv::Mat::zeros(w_pos.rows, w_pos.cols, w_pos.type());

	T* fsrc = w_pos.ptr<T>(0);
	T* fdst = w_neg.ptr<T>(0);
	const unsigned int feature_wp_s = w_pos.step1();
	const unsigned int feature_wn_s = w_neg.step1();

	for (int y = 0; y < w_pos.rows; ++y)
	{
		for (int x = 0; x < w_pos.cols / BINS; ++x)
		{
			T* src = fsrc + y * feature_wp_s + x * BINS;
			T* dst = fdst + y * feature_wn_s + x * BINS;

			for (int z = 0; z < BINS; ++z)
				*(dst++) = -*(src++);
		}
	}

	std::vector<T> v_w_pos((T*) w_pos.data, (T*) w_pos.data + w_pos.total());
	std::vector<T> v_w_neg((T*) w_neg.data, (T*) w_neg.data + w_neg.total());

	typedef typename std::vector<T>::iterator vector_it;
	vector_it pos_max = max_element(v_w_pos.begin(), v_w_pos.end());
	vector_it pos_min = min_element(v_w_pos.begin(), v_w_pos.end());
	vector_it neg_max = max_element(v_w_neg.begin(), v_w_neg.end());

	T scale = std::max(*pos_max, *neg_max);
	T factor = 255 / scale;

	const cv::Mat pos = createVisualization(w_pos, cell_size);
	cv::Mat pos_im;
	pos.convertTo(pos_im, CV_8UC3, factor);

	if (*pos_min < 0)
	{
		const cv::Mat neg = createVisualization(w_neg, cell_size);
		cv::Mat neg_im;
		neg.convertTo(neg_im, CV_8UC3, factor);

		hconcat(pos_im, neg_im, pos_im);
	}

	return pos_im;
}

/**
 * Calculate pixel gradients dx,dy and intensity/magnitude v at point p for the given image
 *
 * @return
 */
template<typename T>
void FeatureHOG<T>::getGradientData(const cv::Point& p, const cv::Mat &image, Gradient &g)
{
	bool failed = true;

	if (image.channels() == 1)
	{
		// Calculate v based on the gray intensity

		if (image.type() == CV_8U)
		{
			std::vector<uchar> pixels;
			getSurroundingPixels(p, (cv::Mat_<uchar>) image, pixels);
			getMagnitudeMono(pixels, g);
			failed = false;
		}
		else if (image.type() == CV_32F)
		{
			std::vector<float> pixels;
			getSurroundingPixels(p, (cv::Mat_<float>) image, pixels);
			getMagnitudeMono(pixels, g);
			failed = false;
		}
		else if (image.type() == CV_64F)
		{
			std::vector<double> pixels;
			getSurroundingPixels(p, (cv::Mat_<double>) image, pixels);
			getMagnitudeMono(pixels, g);
			failed = false;
		}
	}
	else if (image.channels() == 3)
	{
		// Calculate v based on the most dominant color channel

		if (image.type() == CV_8UC3)
		{
			std::vector<cv::Vec3b> pixels;
			getSurroundingPixels(p, (cv::Mat_<cv::Vec3b>) image, pixels);
			getMagnitudeColor(pixels, g);
			failed = false;
		}
		else if (image.type() == CV_32FC3)
		{
			std::vector<cv::Vec3f> pixels;
			getSurroundingPixels(p, (cv::Mat_<cv::Vec3f>) image, pixels);
			getMagnitudeColor(pixels, g);
			failed = false;
		}
		else if (image.type() == CV_64FC3)
		{
			std::vector<cv::Vec3d> pixels;
			getSurroundingPixels(p, (cv::Mat_<cv::Vec3d>) image, pixels);
			getMagnitudeColor(pixels, g);
			failed = false;
		}
	}

	assert(g.dx > -256 && g.dy > -256 && g.dx < 256 && g.dy < 256);
}

/**
 * Return the four pixels (N,S,E,W) surrounding the given point
 *
 * @return
 */
template<typename T> template<typename IT>
void FeatureHOG<T>::getSurroundingPixels(const cv::Point &p, const cv::Mat_<IT> &image, std::vector<IT> &pixels)
{
	cv::Mat p_img = (cv::Mat) image;
	assert(!p_img.empty());

	pixels.resize(4);
	pixels[SurN] = *p_img.ptr<IT>(p.y - 1, p.x);
	pixels[SurS] = *p_img.ptr<IT>(p.y + 1, p.x);
	pixels[SurE] = *p_img.ptr<IT>(p.y, p.x + 1);
	pixels[SurW] = *p_img.ptr<IT>(p.y, p.x - 1);
}

/**
 * Get gradient magnitude (mono)
 *
 * @return
 */
template<typename T> template<typename U>
void FeatureHOG<T>::getMagnitudeMono(const std::vector<U> &pixels, Gradient &g)
{
	assert(!pixels.empty());

	g.dx = (T) pixels[SurE] - (T) pixels[SurW];
	g.dy = (T) pixels[SurS] - (T) pixels[SurN];
	g.v = sqrt(square(g.dx) + square(g.dy));
}

/**
 * Get gradient magnitude (color)
 *
 * @return
 */
template<typename T> template<typename U>
void FeatureHOG<T>::getMagnitudeColor(const std::vector<U> &pixels, Gradient &g)
{
	assert(!pixels.empty());

	g.v = -std::numeric_limits<T>::max();
	for (int c = 0; c < pixels.front().channels; ++c)
	{
		Gradient h;
		h.dx = (T) pixels[SurE][c] - (T) pixels[SurW][c];
		h.dy = (T) pixels[SurS][c] - (T) pixels[SurN][c];
		h.v = (T) sqrt(square(h.dx) + square(h.dy));

		if(h.v > g.v) g = h;
	}
}

/**
 * Flip all the features horizontally of a HOG template using the flipping LUT
 *
 * @return
 */
template<typename T>
void FeatureHOG<T>::flipFeatures(cv::Mat &features)
{
	cv::Mat f_t(features.clone());

	T* fsrc = features.ptr<T>(0);
	T* fdst = f_t.ptr<T>(0);
	const unsigned int featstride = features.step1();

	for (int y = 0; y < features.rows; ++y)
	{
		for (int x = features.cols - 1; x > -1; --x)
		{
			T* src = fsrc + y * featstride + x * DEPTH;
			T* dst = fdst + y * featstride + ((features.cols - 1) - x) * DEPTH;

			for (int z = 0; z < DEPTH; ++z)
				*(dst + LUT_FLIPPED_PERMUTATIONS[z]) = *(src++);
		}
	}
}

/**
 * Fold the constrast sensitive and constrast insensitive features together
 *
 * @return
 */
template<typename T>
const cv::Mat FeatureHOG<T>::fold(const cv::Mat &features)
{
	const int width = features.cols / DEPTH;
	cv::Mat f = cv::Mat::zeros(features.rows, width * BINS, features.type());

	const T* fsrc = features.ptr<T>(0);
	T* fdst = f.ptr<T>(0);
	const unsigned int src_s = features.step1();
	const unsigned int dst_s = f.step1();

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < features.rows; ++y)
		{
			const T* src = fsrc + y * src_s + x * DEPTH;
			T* dst = fdst + y * dst_s + x * BINS;

			T* dst_s = dst;
			for (int z = 0; z < BINS; ++z)
				*(dst_s++) += std::max(*(src++), (T) 0);

			dst_s = dst;
			for (int z = BINS; z < BINS * 2; ++z)
				*(dst_s++) += std::max(*(src++), (T) 0);

			dst_s = dst;
			for (int z = BINS * 2; z < BINS * 3; ++z)
				*(dst_s++) += std::max(*(src++), (T) 0);
		}
	}

	return f;
}

/**
 * generate visualization from a feature vector with given 2D size
 *
 * @return
 */
template<typename T>
const cv::Mat FeatureHOG<T>::visualize(const std::vector<T> &features, const cv::Size &size, int cell_size)
{
	cv::Mat m_features(size.height, size.width, cv::DataType<T>::type);
	m_features.data = (uchar*) &features[0];
	return visualize(m_features, cell_size);
}

/**
 * generate actual visualization using rotated bin images
 *
 * @return
 */
template<typename T>
const cv::Mat FeatureHOG<T>::createVisualization(const cv::Mat &features, const int cell_size)
{
	std::vector<cv::Mat> bins;
	cv::Mat bin = cv::Mat::zeros(cell_size, cell_size, CV_32FC3);

	assert(cell_size % 2 == 0);
	line(bin, cv::Point(cell_size / 2, 0), cv::Point(cell_size / 2, cell_size), cv::Scalar::all(1), 2, CV_AA);

	double angle = 180 / (double) BINS;

	bins.push_back(bin);
	for (int i = 1; i < BINS; ++i)
		bins.push_back(rotateImage(bin, -i * angle));

	cv::Mat image = cv::Mat::zeros(cell_size * features.rows, cell_size * (features.cols / BINS), CV_32FC3);

	const unsigned int feature_s = features.step1();
	const T* fsrc = features.ptr<T>(0);

	for (int y = 0; y < features.rows; y++)
	{
		int ys = y * cell_size;

		for (int x = 0; x < features.cols / BINS; x++)
		{
			const T* src = fsrc + y * feature_s + x * BINS;

			int xs = x * cell_size;
			cv::Mat block = image(cv::Rect(xs, ys, cell_size, cell_size));

			for (int bin = 0; bin < BINS; bin++)
				block = block + (bins.at(bin) * std::max(*(src++), (T) 0));
		}
	}

	return image;
}

/**
 * Rotate an image with a line a certain angle
 *
 * @return
 */
template<typename T>
cv::Mat FeatureHOG<T>::rotateImage(const cv::Mat& source, double angle)
{
	cv::Point2f src_center(source.cols / 2.f, source.rows / 2.f);
	cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	cv::Mat dst;
	warpAffine(source, dst, rot_mat, source.size());
	return dst;
}

/**
 * unit vector used to compute gradient orientation
 *
 * @return
 */
template<typename T>
inline const std::vector<T> FeatureHOG<T>::calcCosLUT()
{
	std::vector<T> cos_a;
	cos_a.reserve(BINS);
	const T angle = (T) CV_PI / (T) BINS;
	for (int o = 0; o < BINS; ++o)
		cos_a.emplace_back(cos(o * angle));
	return cos_a;
}

/**
 * unit vector used to compute gradient orientation
 *
 * @return
 */
template<typename T>
inline const std::vector<T> FeatureHOG<T>::calcSinLUT()
{
	std::vector<T> sin_a;
	sin_a.reserve(BINS);
	const T angle = (T) CV_PI / (T) BINS;
	for (int o = 0; o < BINS; ++o)
		sin_a.emplace_back(sin(o * angle));
	return sin_a;
}

/**
 * calculate the mappings for horizontally flipping of a HOG template
 *
 * @return
 */
template<typename T>
std::vector<int> FeatureHOG<T>::calcPermutations()
{
	std::vector<int> permutations = std::vector<int>(DEPTH);

	for (int o = 0; o < BINS; ++o)
	{
		permutations[o] = BINS - o;
		permutations[o + BINS] = 2 * BINS - o == 2 * BINS ? 0 : 2 * BINS - o;
		permutations[o + 2 * BINS] = 3 * BINS - o == 3 * BINS ? 2 * BINS : 3 * BINS - o;
	}

	if (USE_TEXTURE_FEATURES)
	{
		permutations[3 * BINS + 0] = 3 * BINS + 2;
		permutations[3 * BINS + 1] = 3 * BINS + 3;
		permutations[3 * BINS + 2] = 3 * BINS + 0;
		permutations[3 * BINS + 3] = 3 * BINS + 1;
	}

	if (USE_OCCLUSION_FEATURE)
	{
		permutations[3 * BINS + 4] = 3 * BINS + 4;
	}

	return permutations;
}

template class FeatureHOG<float> ;
template class FeatureHOG<double> ;

// end ************************************ Felzenszwalb HOG ******************************************* end

} /* namespace nl_uu_science_gmt */
