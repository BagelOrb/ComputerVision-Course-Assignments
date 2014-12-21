#pragma once

#include <algorithm> // transform
#include <opencv2/opencv.hpp>

#include <opencv2/ml/ml.hpp>


using namespace cv;

//#include "General.h"
//using namespace nl_uu_science_gmt; 
struct MixtureColorModel_TrainData
{
	Mat a;
	Mat b;
	Mat c;
	Mat d;
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

class MixtureColorModel
{
public:
	MixtureColorModel();
	~MixtureColorModel();
	void generateModels();
	void saveModels();
	void saveModels(string filename);
	bool load();
	bool load(string filename);
	void test();
private:
	static const int covMatType = EM::COV_MAT_GENERIC; //EM::COV_MAT_DIAGONAL
	EM* emA;
	EM* emB;
	EM* emC;
	EM* emD;

	static void saveModel(EM& em, FileStorage& fs, std::string prefix);
	static EM* loadModel(FileStorage& fs, std::string prefix, Mat& data);

};

