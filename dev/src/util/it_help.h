#pragma once

#include "../inc.h"

double get_max(ITensor&);
double get_max_abs(ITensor&);
double get_sum(ITensor&);

ITensor trivialIT();
void show_shape(const ITensor&);
void show_shape(const IndexSet&);


std::string myprint_ts(const TagSet& ts);
std::string myprint_idx(const Index& idx);
std::string myprint_is(const IndexSet& is);
std::string myprint(const ITensor& A);
std::vector<double> mypeek(const ITensor& A);

vector_no_init<Real>* peek(const ITensor& A);
vector_no_init<Cplx>* peek_cx(const ITensor& A);




//simultaneous sort vectors A and B according to descending order in A
template <typename A, typename B>
void ssort(std::vector<A>& vecA, std::vector<B>& vecB) {

	long lenA = vecA.size();
	long lenB = vecB.size();
	if (lenA != lenB) Error("size don't match!");

	using AB = std::pair<A, B>;

	std::vector<AB> V = {};
	V.reserve(lenA);
	for (long k = 0; k < lenA; k++) {
		V.push_back(std::make_pair(vecA[k], vecB[k]));
	}

	auto larger = [](const AB& a, const AB& b) {
		return std::abs(a.first) > std::abs(b.first);
	};

	sort(V.begin(), V.end(), larger);

	for (long k = 0; k < lenA; k++) {
		vecA[k] = V[k].first;
		vecB[k] = V[k].second;
	}
}