#pragma once

#include <boost/random/discrete_distribution.hpp>
#include <chrono> // randomized seed based on time
#include <random>

using namespace std;


/*
class ParamState
{
public:
virtual ParamState* getNearbyRandom(double std_dev, std::default_random_engine& gen);
};
*/

/*
template<class ParamStateType>
class PerformanceEvaluator_Base
{
public:
virtual double evaluate(ParamStateType& params);
};
*/

/*!
A heuristic search algorithm similar to beam-search.

Beam search:
	- at each epoch
		- expand on the current parameter states (sample new ones)
		- prune all parameter states to a number of beam-width (keep the [beam-width] best ones)

Mixture Model Beam Search expands on the current set of parameter states by sampling a parameter state from the existing ones, 
based on the performance of the parameter states.
It does this by applying a discrete distribution over the parameter settings, where the probabilities are proportional to the performances.

The sampled parameter state is then used to sample a new nearby parameter state, based on the standard deviation at the current epoch.
The standard deviation is set by hand by MMBeamSearch, instead of fitted to the parameter states.
The standard deviation is reduced exponentially after each epoch, so that we sample closer and closer to better and better points in the parameter space.

We suppose the ParamstateType contains a function 
\beginverbatim
HSV_State* getNearbyRandom(double std_dev, std::default_random_engine& gen)
\endverbatim
which samples a new ParamstateType, with a given standard deviation, which it uses homogeneously over each dimensino in the parameter space.
In case this function uses a Gaussion distribution, each Mixture Model Beam Search epoch can be seen as a two step process:
	- sampling new parameter states from a gaussian mixture model
	- pruning the total parameter states

*/
template<class ParamstateType, class PerformanceEvaluator>
class MMBeamSearch
{
	//! A simple construct pairing a parameter state with its performance
	struct Result 
	{
		ParamstateType* params; //!< the point in parameter space
		double performance; //!< the performance of the point in parameter space
		Result(ParamstateType* s, double p) : params(s), performance(p) {};
		~Result()
		{
			//delete params;
		};

		//! The highest performance result is first
		inline bool operator< (Result &b){
			return performance > b.performance;
		}
	};

	//! The state of the current epoch in the Mixture Model Beam Search algorithm
	struct State
	{
		//! all results currently tracked
		vector<Result> bestN;	

		//! sample random result from a discrete distribution over all current results
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

	unsigned seed; //!< seed used to initialize the random number generator
	default_random_engine generator; //!< ranodm number generator

public:

	int beam_size; //!< number of results after each epoch

	int sample_size; //!< number of points newly sampled iun each epoch

	int max_depth; //!< maximal depth, number of epochs

	PerformanceEvaluator evaluator; //!< evaluator for computing the performance of a given parameter setting

	/*!
	Construction of a Mixture Model Beam Search algorithm object.

	Initializes the random number generator.

	@param beam_size_ the number of results after pruning at the end of each epoch
	@param sample_size_ the number of results newly sampled during each epoch
	@param max_depth_ the maximal depth, number of epochs
	@param evaluator_ the evaluator for computing the performance of a given parameter setting
	*/
	MMBeamSearch(int beam_size_, int sample_size_, int max_depth_, PerformanceEvaluator evaluator_)
		: beam_size(beam_size_) 
		, sample_size(sample_size_)
		, max_depth(max_depth_)
		, evaluator(evaluator_)
	{
		seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
		generator = default_random_engine(seed);
	};

	State state;//!< the current state of the mixture model beam search algorithm

	/*!
	Initialize the parameter states from which to start searching.
	*/
	void initialize(vector<ParamstateType*> items)
	{
		typedef MMBeamSearch<ParamstateType, PerformanceEvaluator>::Result Result;
		for (ParamstateType* item : items)
		{
			state.bestN.push_back(Result(item, evaluator.evaluate(*item)));
		}

	}

	/*!
	Perform the beam search algorithm.

	@param std_dev_start the initial 
	*/
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