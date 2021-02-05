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

	IndexSet active_inds() {
		// the unprimed indices of A are those of vector x
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

//! A lightweight array for POD types. For internal use only!
template<typename eT>
class itpodarray
{
public:

	arma_aligned const uword n_elem; //!< number of elements held
	arma_aligned       eT* mem;    //!< pointer to memory used by the object


protected:
	//! internal memory, to avoid calling the 'new' operator for small amounts of memory.
	//arma_align_mem eT mem_local[podarray_prealloc_n_elem::val];


public:

	 ~itpodarray();
	  itpodarray();

	//inline                 itpodarray(const itpodarray& x);
	//inline const itpodarray& operator=(const itpodarray& x);

	//arma_inline explicit itpodarray(const uword new_N);
	//
	//arma_inline explicit itpodarray(const eT* X, const uword new_N);

	// template<typename T1>
	// inline explicit podarray(const Proxy<T1>& P);

	arma_inline eT& operator[] (const uword i);
	arma_inline eT  operator[] (const uword i) const;

	arma_inline eT& operator() (const uword i);
	arma_inline eT  operator() (const uword i) const;

	inline void set_min_size(const uword min_n_elem);

	inline void set_size(const uword new_n_elem);
	inline void reset();


	inline void fill(const eT val);

	inline void zeros();
	inline void zeros(const uword new_n_elem);

	arma_inline       eT* memptr();
	arma_inline const eT* memptr() const;

	arma_hot inline void copy_row(const arma::Mat<eT>& A, const uword row);


protected:

	inline void init_cold(const uword new_n_elem);
	inline void init_warm(const uword new_n_elem);
};



}