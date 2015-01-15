/*
 * MySVM.h
 *
 *  Created on: 27 aug. 2013
 *      Author: Coert
 */

#ifndef MYSVM_H_
#define MYSVM_H_

#include "opencv2/ml/ml.hpp"

namespace nl_uu_science_gmt {

class MySVM: public CvSVM {
	bool is_trained;

public:
	MySVM();
	virtual ~MySVM();

	void run(const cv::Mat &, const cv::Mat &, const cv::SVMParams &);

	CvSVMDecisionFunc* getDecisionFunc()
	{
		return decision_func;
	}

	CvSVMSolver* getSolver()
	{
		return solver;
	}

	CvSVMKernel* getKernel()
	{
		return kernel;
	}

	bool isTrained() const
	{
		return is_trained;
	}
};

} /* namespace nl_uu_science_gmt */
#endif /* MYSVM_H_ */
