#pragma once

#include "pch.h"
#include "it_help.h"


//T can be double or complex, determined by the element type of A
//template<typename T>
class ITensorMap
{
	ITensor const& A_;
	mutable long size_;
	
	//IndexSet is;
	//vector_no_init<T>* dptr;

public:

	ITensorMap(ITensor const& A)
		: A_(A)
	{
		size_ = 1;
		int psize_ = 1;
		for (auto& I : A_.inds())
		{
			if (I.primeLevel() == 0)
				size_ *= dim(I);
			if (I.primeLevel() == 1)
				psize_ *= dim(I);
		}
		// make sure A is a square matrix
		if (size_ != psize_) {
			itensor::error("ITensorMap unprimed and primed dim don't match.\n");
		}
	}

	// the unprimed indices of A are those of vector x
	IndexSet active_inds() const{
		return findInds(A_, "0");
	}

	//define max-vec product A x -> b
	void
		product(ITensor const& x, ITensor& b) const
	{
		b = A_ * x;
		b.noPrime();
	}

	//dim of this matrix
	long
		size() const
	{
		return size_;
	}

};


namespace itwrap {

bool
eig_arpack(std::vector<Cplx>& eigval, std::vector<ITensor>& eigvecs, const ITensorMap& AMap, Args const& args);

}