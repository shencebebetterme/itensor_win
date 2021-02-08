#include "it_help.h"
#include <algorithm>


// max elements of an real ITensor
double get_max(ITensor& A) {

	if (!isReal(A)) return 0;

	std::vector<double> vec;
	for (auto it : iterInds(A)) {
		vec.push_back(A.real(it));
	}
	
	return *max_element(vec.begin(), vec.end());
}


double get_max_abs(ITensor& A) {
	std::vector<double> vec;
	for (auto it : iterInds(A)) {
		vec.push_back(abs(A.real(it)));
	}

	return *max_element(vec.begin(), vec.end());
}


// sum of elements of an real ITensor
double get_sum(ITensor& A) {
	double sum = 0;

	for (auto it : iterInds(A)) {
		sum += A.real(it);
	}

	return sum;
}


// return an ITensor with 0 indices and norm 1;
ITensor trivialIT() {
	Index i(1);
	ITensor A = setElt(i = 1);
	return (A * A);
}


void show_shape(const ITensor& A) {
	show_shape(A.inds());
}

void show_shape(const IndexSet& is) {
	printf(" [");
	for (auto i : is) {
		printf(" %d,", dim(i));
	}
	printf("] ");
}




std::string myprint_ts(const TagSet& ts)
{
	std::string s;
	s = "''";
	s += str(primeLevel(ts)) + ",";
	for (auto i : range(size(ts)))
	{
		s += ts[i];
		if (i < (size(ts) - 1)) s += ",";
	}
	s += "''";
	return s;
}

std::string myprint_idx(const Index& idx) {
	string s = myprint_ts(idx.tags());
	s += " dim=" + str(idx.dim());
	s += " id=" + str(id(idx) % 1000);
	s += " || ";
	return s;
}

std::string myprint_is(const IndexSet& is) {
	string s;
	for (auto idx : is) {
		s += myprint_idx(idx);
	}
	return s;
}

std::string myprint(const ITensor& A) {
	return myprint_is(A.inds());
	//return A.myprint();
}

std::vector<double> mypeek(const ITensor& A) {
	int count = 0;
	constexpr int count_max = 10;
	std::vector<double> vec = {};

	for (auto it : iterInds(A)) {
		vec.push_back(A.elt(it));
		count += 1;
		if (count == 10) break;
	}
	return vec;
}

vector_no_init<Real>* peek(const ITensor& A) {
	if (!isReal(A)) {
		Error("Tensor not real!");
	}
	vector_no_init<Real>* sP = &((*((ITWrap<Dense<double>>*) & (*A.store()))).d.store);
	return sP;
}

vector_no_init<Cplx>* peek_cx(const ITensor& A) {
	if (!isComplex(A)) {
		Error("Tensor not complex!");
	}
	vector_no_init<Cplx>* sP = &((*((ITWrap<Dense<Cplx>>*) & (*A.store()))).d.store);
	return sP;
}









