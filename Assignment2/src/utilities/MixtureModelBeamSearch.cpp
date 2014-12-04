#include "MixtureModelBeamSearch.h"

#include <algorithm> // sort

/*
template<class ParamstateType, class PerformanceEvaluator>
typename MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result MMBeamSearch<ParamstateType, PerformanceEvaluator>::State::sampleRandomResult()
{
	typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;
	vector<double> perfs;
	for (Result r : bestN)
		perfs.push_back(r.performance);

	discrete_distribution<int> dist(perfs.begin(), perfs.end());

	default_random_engine generator;

	return bestN[dist(generator)];
}

template<class ParamstateType, class PerformanceEvaluator>
void MMBeamSearch<ParamstateType, PerformanceEvaluator>::initialize(vector<ParamstateType*> items)
{
	typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;
	for (ParamstateType* item : items)
	{
		bestN.push_back(Result(item, PerformanceEvaluator.evaluate(*item)));
	}

}


template<class ParamstateType, class PerformanceEvaluator>
typename MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result MMBeamSearch<ParamstateType, PerformanceEvaluator>::perform(double std_dev_start, double std_dev_decrement_factor)
{
	double std_dev = std_dev_start;
	for (int depth = 1; deptch < max_depth; depth++)
	{
		getNewSamples(std_dev);
		purgeResults(std_dev);

		std_dev *= std_dev_decrement_factor;
	}

	return bestN[0];
}




template<class ParamstateType, class PerformanceEvaluator>
void MMBeamSearch<ParamstateType, PerformanceEvaluator>::getNewSamples(double std_dev)
{
	typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;

	vector<Results> new_results;
	for (int i = 0; i < sample_size; i++)
	{
		Result r = state.sampleRandomResult();
		ParamstateType* new_params = r.params.getNearbyRandom(std_dev, generator);
		double performance = PerformanceEvaluator.evaluate(*new_params);
		new_results.push_back(Result(new_params, performance));
	}

	state.bestN.insert(bestN.end(), new_results.begin(), new_results.end());

}



template<class ParamstateType, class PerformanceEvaluator>
bool compareResults(typename MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result& a, typename MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result& b) { return a.performance > b.performance; }

template<class ParamstateType, class PerformanceEvaluator>
void MMBeamSearch<ParamstateType, PerformanceEvaluator>::purgeResults()
{
	sort(state.bestN.begin(), state.bestN.end(), compareResults);
	state.bestN.erase(state.bestN.begin() + beam_size, state.bestN.end());
}

*/
