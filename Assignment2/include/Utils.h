#pragma once

using namespace std;


class Utils
{
public:
	template <class InputIterator, class OutputIterator, class Func>
	static OutputIterator transform(InputIterator first1, InputIterator last1, InputIterator first2,
		OutputIterator result, Func op) // note: different order of arguments than the rest (same order as std::transform)
	{
		while (first1 != last1) {
			*result = op(*first1, *first2++);
			++result; ++first1;
		}
		return result;
	}

	template <class InputIterator, class OutputIterator, class Func>
	static OutputIterator transform(InputIterator first1, InputIterator last1,
		OutputIterator result, Func op, InputIterator first2, InputIterator first3)
	{
		while (first1 != last1) {
			*result = op(*first1, *first2++, *first3++);
			++result; ++first1;
		}
		return result;
	}

	template <class InputIterator, class OutputIterator, class Func>
	static OutputIterator transform(InputIterator first1, InputIterator last1,
		OutputIterator result, Func op, InputIterator first2, InputIterator first3, InputIterator first4)
	{
		while (first1 != last1) {
			*result = op(*first1, *first2++, *first3++, *first4++);
			++result; ++first1;
		}
		return result;
	}
	template <class InputIterator, class OutputIterator, class Func>
	static OutputIterator transform(InputIterator first1, InputIterator last1,
		OutputIterator result, Func op, InputIterator first2, InputIterator first3, InputIterator first4, InputIterator first5)
	{
		while (first1 != last1) {
			*result = op(*first1, *first2++, *first3++, *first4++, *first5++);
			++result; ++first1;
		}
		return result;
	}
	template <class InputIterator, class OutputIterator, class Func>
	static OutputIterator transform(InputIterator first1, InputIterator last1,
		OutputIterator result, Func op, InputIterator first2, InputIterator first3, InputIterator first4, InputIterator first5, InputIterator first6)
	{
		while (first1 != last1) {
			*result = op(*first1, *first2++, *first3++, *first4++, *first5++, *first6++);
			++result; ++first1;
		}
		return result;
	}
};

