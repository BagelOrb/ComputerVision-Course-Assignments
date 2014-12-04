#pragma once
//#include <opencv2/opencv.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <chrono> // randomized seed based on time
#include <random>
//#include <cstdlib> // rand()
//#include <ctime> // time

using namespace std;
//using namespace cv;

/*
class ParamState
{
public:
	virtual ParamState* getNearbyRandom(double std_dev);
};

template<class ParamStateType>
class PerformanceEvaluator_Base
{
public:
	virtual double evaluate(ParamStateType& params);
};
*/


template<class Result>
bool compareResults(Result& a, Result& b) { return a.performance > b.performance; }



template<class ParamstateType, class PerformanceEvaluator>
class MMBeamSearch
{
	struct Result 
	{
		ParamstateType* params;
		double performance;
		Result(ParamstateType* s, double p) : params(s), performance(p) {};
		~Result()
		{
			//delete params;
		};
		inline bool operator< (Result &b){
			return performance > b.performance;
		}
	};

	struct State
	{
		vector<Result> bestN;	

		Result sampleRandomResult() // sample Result from the mixture of results
		{
			typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;
			vector<double> perfs;
			for (Result r : bestN)
				perfs.push_back(r.performance);

			boost::random::discrete_distribution<int> dist(perfs.begin(), perfs.end());

			default_random_engine generator;

			return bestN[dist(generator)];
		}
	};

	unsigned seed;
	default_random_engine generator;

public:

	int beam_size; // number of results after each epoch

	int sample_size; // how many points are evaluated before downsizing to the beam_size

	int max_depth; // max recursion depth

	PerformanceEvaluator evaluator;

	MMBeamSearch(int beam_size_, int sample_size_, int max_depth_, PerformanceEvaluator evaluator_)
		: beam_size(beam_size_) 
		, sample_size(sample_size_)
		, max_depth(max_depth_)
		, evaluator(evaluator_)
	{
		seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
		generator = default_random_engine(seed);
	};

	State state;

	void initialize(vector<ParamstateType*> items)
	{
		typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;
		for (ParamstateType* item : items)
		{
			state.bestN.push_back(Result(item, evaluator.evaluate(*item)));
		}

	}


	Result perform(double std_dev_start, double std_dev_decrement_factor)
	{
		double std_dev = std_dev_start;
		for (int depth = 0; depth < max_depth; depth++)
		{
			getNewSamples(std_dev);

			purgeResults();

			std_dev *= std_dev_decrement_factor;

			cout << "purged at depth "<< depth <<":" << endl;
			for (Result result : state.bestN)
				cout << static_cast<int>(result.params->h) << ", \t" << static_cast<int>(result.params->s) << ", \t" << static_cast<int>(result.params->v) << ": \t" << result.performance << endl;

		}

//		cout << "all eventual:" << endl;
//		for (Result result : state.bestN)
//			cout << static_cast<int>(result.params->h) << ", \t" << static_cast<int>(result.params->s) << ", \t" << static_cast<int>(result.params->v) << ": \t" << result.performance << endl;



		return state.bestN[0];
	}

protected:
	//void getFirstSamples(); // uniform over parameter space
	void getNewSamples(double std_dev)
	{
		typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;

		vector<Result> new_results;
		for (int i = 0; i < sample_size; i++)
		{
			Result r = state.sampleRandomResult();
			ParamstateType* new_params = r.params->getNearbyRandom(std_dev, generator);
			double performance = evaluator.evaluate(*new_params);
			new_results.push_back(Result(new_params, performance));
		}

		state.bestN.insert(state.bestN.end(), new_results.begin(), new_results.end());

	}

private:
	//! used in purgeResults()
	//bool compareResults(Result& a, Result& b) { return a.performance > b.performance; };

protected:
	void purgeResults()
	{
		std::sort(state.bestN.begin(), state.bestN.end());// , &compareResults);
		state.bestN.erase(state.bestN.begin() + beam_size, state.bestN.end());
	}


};