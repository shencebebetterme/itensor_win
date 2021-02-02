#pragma once

#include "itensor/all_basic.h"
#include "itensor/util/print_macro.h"
using namespace itensor;



class ITensorMap
{
	ITensor const& A_;
	mutable long size_;

public:

	ITensorMap(ITensor const& A)
		: A_(A)
	{
		size_ = 1;
		for (auto& I : A_.inds())
		{
			if (I.primeLevel() == 0)
				size_ *= dim(I);
		}
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


Complex
my_arnoldi(const ITensorMap& A,
	ITensor& vec,
	Args const& args);

std::vector<Complex>
my_arnoldi(const ITensorMap& A,
	std::vector<ITensor>& phi,
	Args const& args);