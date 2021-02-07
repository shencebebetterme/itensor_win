#pragma once

#include "pch.h"
#include "it_help.h"

//todo: what if A_ is not an ITensor
//T can be double or complex, determined by the element type of A
//template<typename T>
class ITensorMap
{
	using T = double;

	ITensor const& A_;
	mutable long size_;
	IndexSet act_is;

	ITensor itx;
	ITensor ity;

	
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

		// initialize itx and ity
		act_is = active_inds();
		itx = ITensor(act_is);
		ity = ITensor(act_is);

		Cplx val(1.0, 1.0);
		itx.set(*iterInds(act_is), *(T*)&val);
		ity.set(*iterInds(act_is), *(T*)&val);
		
	}

	// the unprimed indices of A are those of vector x
	IndexSet active_inds() const{
		return findInds(A_, "0");
	}


	// A lower level interface of linear operator, directly
	// act on the arpack memory
	// if vec := *bin to *(bin+n)
	// then replace *bout to *(bout+n) by A*vec
	void Amul(T* pin, T* pout, int n) {

		//todo: avoid this memory copying by modifying the allocator of vector_no_init
		// copy the memory to itensor itx, do contraction, then copy memory of ity back
		vector_no_init<T>& dvecx = (*((ITWrap<Dense<T>>*) & (*itx.store()))).d.store;
		dvecx.assign(pin, pin + n);
		this->product(itx, ity);
		vector_no_init<T>& dvecy = (*((ITWrap<Dense<T>>*) & (*ity.store()))).d.store;
		std::memcpy(pout, &dvecy[0], n * sizeof(T));
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

//template<typename T>
bool
eig_arpack(std::vector<Cplx>& eigval, std::vector<ITensor>& eigvecs, const ITensorMap& AMap, Args const& args);

}