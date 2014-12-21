#pragma once

#include <algorithm> // transform
#include <opencv2/opencv.hpp>

#include <opencv2/ml/ml.hpp>


using namespace cv;

//#include "General.h"
//using namespace nl_uu_science_gmt;

/*!
A struct for handling the training data used for the generation of the gaussian mixture color models.

A file is used in which each pixel of a given frame is coloured according to which person the person belongs to, and black otehrwise.
Note that only a single frame of a single camera has been used.
*/
struct MixtureColorModel_TrainData 
{
	Mat a; //!< the color data of person 1
	Mat b; //!< the color data of person 2
	Mat c; //!< the color data of person 3
	Mat d; //!< the color data of person 4

	//! Reads the training data from file
	MixtureColorModel_TrainData() 
	{
		cv::Mat foreground = cv::imread("data\\cam1frame752-manualforeground-colored.bmp");
		cv::Mat frame = cv::imread("data\\cam1frame752-original.jpg");

		std::vector<Vec3b> colorsA;
		std::vector<Vec3b> colorsB;
		std::vector<Vec3b> colorsC;
		std::vector<Vec3b> colorsD;
		{
			MatIterator_<Vec3b> class_pixel = foreground.begin<Vec3b>();
			for (MatIterator_<Vec3b> pixel = frame.begin<Vec3b>(); pixel != frame.end<Vec3b>(); pixel++)
			{
				if ((*class_pixel)[0] > 250) //blue
					colorsA.push_back(*pixel);
				else if ((*class_pixel)[1] > 250) //red
					colorsB.push_back(*pixel);
				else if ((*class_pixel)[2] > 250) //green
					colorsC.push_back(*pixel);
				else if ((*class_pixel)[0] > 120) //gray
					colorsD.push_back(*pixel);
				class_pixel++;
			}
		}

		cv::Mat colorImgA(colorsA, true);
		a = colorImgA.reshape(1, colorImgA.cols * colorImgA.rows);
		cv::Mat colorImgB(colorsB, true);
		b = colorImgB.reshape(1, colorImgB.cols * colorImgB.rows);
		cv::Mat colorImgC(colorsC, true);
		c = colorImgC.reshape(1, colorImgC.cols * colorImgC.rows);
		cv::Mat colorImgD(colorsD, true);
		d = colorImgD.reshape(1, colorImgD.cols * colorImgD.rows);


	};
};

/*!
Gaussian mixture models used to model the color palettes of four people.
*/
class MixtureColorModel
{
public:
	MixtureColorModel(); //!< does no initialization
	~MixtureColorModel();
	void generateModels(); //!< generate models based on manually labelled camera images
	void saveModels(); //!< save the internal state to mixture_color_models.yml
	void saveModels(string filename); //!< save the internal state
	bool load();//!< load the internal state from mixture_color_models.yml
	bool load(string filename);//!< load the internal state
	void test(); //!< convert each pixel of the original sample frame occording to which color model has the smallest distance and output the image to generated_output.bmp
	double distanceA(Vec3b color) { return 0 - emA->predict(cv::Mat(color, true).t())[0]; }; //!< get the distance of a color to the color model of person A
	double distanceB(Vec3b color) { return 0 - emB->predict(cv::Mat(color, true).t())[0]; }; //!< get the distance of a color to the color model of person B
	double distanceC(Vec3b color) { return 0 - emC->predict(cv::Mat(color, true).t())[0]; }; //!< get the distance of a color to the color model of person C
	double distanceD(Vec3b color) { return 0 - emD->predict(cv::Mat(color, true).t())[0]; }; //!< get the distance of a color to the color model of person D
	//! Get the distance of a color to the color model of a person. The distance is the negated log-likelihood of the probability that the color belongs to the modelling distribution.
	double distance(Vec3b color, uchar colorModel)
	{
		switch (colorModel)
		{
		case 0:
			return distanceA(color);
		case 1:
			return distanceB(color);
		case 2:
			return distanceC(color);
		case 3:
			return distanceD(color);
		default:
			std::cerr << "Cluster number out of bounds! : " << colorModel << std::endl << "Number of clusters is 4." << std::endl;
			return -1e20;
		}
	}
private:
	static const int covMatType = EM::COV_MAT_GENERIC; //EM::COV_MAT_DIAGONAL //!< The restrictions on the coveriance matrix used in the gaussian mixture models
	EM* emA; //!< the Gaussian mixture model of person A
	EM* emB;//!< the Gaussian mixture model of person B
	EM* emC;//!< the Gaussian mixture model of person C
	EM* emD;//!< the Gaussian mixture model of person D

	static void saveModel(EM& em, FileStorage& fs, std::string prefix); //!< Save a single gaussian mixture model to the filestorage
	static EM* loadModel(FileStorage& fs, std::string prefix, Mat& data); //!< Load a single gaussian mixture model from the filestorage

};

