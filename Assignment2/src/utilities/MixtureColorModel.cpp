#include "MixtureColorModel.h"



MixtureColorModel::MixtureColorModel()
{
}


MixtureColorModel::~MixtureColorModel()
{
	delete emA;
	delete emB;
	delete emC;
	delete emD;
}



void MixtureColorModel::saveModels()
{
	saveModels("mixture_color_models.yml");
}
void MixtureColorModel::saveModels(string filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened()) { std::cout << "unable to open file storage!" << std::endl; return; }

	saveModel(*emA, fs, "A-");
	saveModel(*emB, fs, "B-");
	saveModel(*emC, fs, "C-");
	saveModel(*emD, fs, "D-");


	fs.release();

}

void MixtureColorModel::saveModel(EM& em, FileStorage& fs, std::string prefix)
{

	fs << (prefix + "weights") << em.get<Mat>("weights");
	fs << (prefix + "means") << em.get<Mat>("means");
	fs << (prefix + "covs") << em.get<std::vector<Mat>>("covs");

}


void MixtureColorModel::generateModels()
{
	MixtureColorModel_TrainData data;
	

	int maxIters = 100; 
	int n_color_clusters = 4;


	emA = new EM(n_color_clusters, covMatType, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, maxIters, FLT_EPSILON));
	std::cout << "training A..." << std::endl;
	emA->train(data.a);

	emB = new EM(n_color_clusters, covMatType, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, maxIters, FLT_EPSILON));
	std::cout << "training B..." << std::endl;
	emB->train(data.b);

	emC = new EM(n_color_clusters, covMatType, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, maxIters, FLT_EPSILON));
	std::cout << "training C..." << std::endl;
	emC->train(data.c);

	emD = new EM(n_color_clusters, covMatType, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, maxIters, FLT_EPSILON));
	std::cout << "training D..." << std::endl;
	emD->train(data.d);

}
void MixtureColorModel::test()
{

	Mat frame = imread("data\\cam1frame752-original.jpg");

	Mat out(frame); // copy so that output mat has same dimensionality

	{
		MatIterator_<Vec3b> out_pixel = out.begin<Vec3b>();
		for (MatIterator_<Vec3b> pixel = frame.begin<Vec3b>(); pixel != frame.end<Vec3b>(); pixel++)
		{
			double valA = emA->predict(cv::Mat(*pixel, true).t())[0];
			double valB = emB->predict(cv::Mat(*pixel, true).t())[0];
			double valC = emC->predict(cv::Mat(*pixel, true).t())[0];
			double valD = emD->predict(cv::Mat(*pixel, true).t())[0];
			if (valA > valB && valA > valC && valA > valD)
			{
				(*out_pixel)[0] = 255; (*out_pixel)[1] = 0; (*out_pixel)[2] = 0;
			}
			else if (valB > valC && valB > valD)
			{
				(*out_pixel)[0] = 0; (*out_pixel)[1] = 255; (*out_pixel)[2] = 0;
			}
			else if (valC > valD)
			{
				(*out_pixel)[0] = 0; (*out_pixel)[1] = 0; (*out_pixel)[2] = 255;
			}
			else
			{
				(*out_pixel)[0] = 128; (*out_pixel)[1] = 128; (*out_pixel)[2] = 128;
			}

			out_pixel++;
		}
	}


	cv::imwrite("generated_output.bmp", out);

}

bool MixtureColorModel::load()
{
	return load("mixture_color_models.yml");
}
bool MixtureColorModel::load(string filename)
{
	

	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened()) { std::cout << "unable to open file storage!" << std::endl; return false; }

	MixtureColorModel_TrainData data;

	std::cout << "Loading MixtureColorModel A..." << std::endl;
	emA = loadModel(fs, "A-", data.a);
	std::cout << "Loading MixtureColorModel B..." << std::endl;
	emB = loadModel(fs, "B-", data.b);
	std::cout << "Loading MixtureColorModel C..." << std::endl;
	emC = loadModel(fs, "C-", data.c);
	std::cout << "Loading MixtureColorModel D..." << std::endl;
	emD = loadModel(fs, "D-", data.d);

	fs.release();

	return true;
}

EM* MixtureColorModel::loadModel(FileStorage& fs, std::string prefix, Mat& data)
{
	Mat weights;
	fs[prefix + "weights"] >> weights;
	Mat means;
	fs[prefix + "means"] >> means;
	std::vector<Mat> covs;
	fs[prefix + "covs"] >> covs;

	int nclusters = covs.size();
	int maxIters = 1; // EM::DEFAULT_MAX_ITERS
	TermCriteria& termCrit = TermCriteria(TermCriteria::COUNT, maxIters, FLT_EPSILON);
	EM* em = new EM(nclusters, covMatType, termCrit);

	/*
	std::vector<Vec3b> fakeData;
	fakeData.push_back(Vec3b(0, 0, 0));
	fakeData.push_back(Vec3b(1, 0, 0));
	fakeData.push_back(Vec3b(0, 1, 0));
	fakeData.push_back(Vec3b(0, 0, 1));
	fakeData.push_back(Vec3b(0, 1, 1));

	Mat fakeDatas(fakeData);
	fakeDatas = fakeDatas.reshape(1, fakeDatas.rows * fakeDatas.cols);
	//Mat fakeDatas(3, 10, CV_8UC1);
	*/

	em->trainE(data,means,covs,weights);
	return em;
}
