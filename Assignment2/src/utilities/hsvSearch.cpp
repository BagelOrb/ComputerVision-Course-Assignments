#include "hsvSearch.h"


#include "MixtureModelBeamSearch.h"


using namespace std;
//using namespace cv;

/*
HSV_State* HSV_State::getNearbyRandom(double std_dev, std::default_random_engine gen)
{
	std::normal_distribution<double> dist(0, std_dev);
	return new HSV_State(h + dist(gen), s + dist(gen), v + dist(gen));
};



double HSV_Evaluator::evaluate(HSV_State& params)
{
	return params.h * (1 - params.h) + params.s * (1 - params.s) + params.v * (1 - params.v);
};
*/

int main_hsvSearch_test()
{
	MMBeamSearch<HSV_State, HSV_Evaluator_Test> search(10, 50, 20, HSV_Evaluator_Test());
	vector<HSV_State*> initials;

	HSV_State* first = new HSV_State(1,2,3); // delete called by MMBeamSearch::Result

	initials.push_back(first);

	search.initialize(initials);

	auto result = search.perform(100, .8);

	cout << " best result : " << endl;
	cout << static_cast<int>(result.params->h) << ", " << static_cast<int>(result.params->s) << ", " << static_cast<int>(result.params->v ) << ": " << result.performance << endl;

	return 0;
};