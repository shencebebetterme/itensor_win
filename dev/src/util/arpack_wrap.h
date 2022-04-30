#pragma once

#include "inc.h"
#include "it_help.h"

//todo: what if A_ is not an ITensor
//T can be double or complex, determined by the element type of A
//template<typename T>
class ITensorMapBase
{
public:
	//using T = double;

	//ITensor const& A_;
	mutable long size_;
	IndexSet act_is;

	//ITensor itx;
	//ITensor ity;

	
	//IndexSet is;
	//vector_no_init<T>* dptr;

public:

	//ITensorMap(ITensor const& A)
	//	: A_(A)
	//{
	//	size_ = 1;
	//	int psize_ = 1;
	//	for (auto& I : A_.inds())
	//	{
	//		if (I.primeLevel() == 0)
	//			size_ *= dim(I);
	//		if (I.primeLevel() == 1)
	//			psize_ *= dim(I);
	//	}
	//	// make sure A is a square matrix
	//	if (size_ != psize_) {
	//		itensor::error("ITensorMap unprimed and primed dim don't match.\n");
	//	}

	//	act_is = findInds(A, "0");
	//}


	ITensorMapBase(IndexSet& is) 
		//:A_(ITensor())
	{
		size_ = 1;
		for (auto& I : is) {
			//if (I.primeLevel() > 0) Error("Index not primed.");
			size_ *= dim(I);
		}
		act_is = is;
	}


	// the unprimed indices of A are those of vector x
	IndexSet active_inds() const{
		//return findInds(A_, "0");
		return act_is;
	}

	//dim of this matrix
	long size() const
	{
		return size_;
	}


	// A lower level interface of linear operator, directly
	// act on the arpack memory
	// if vec := *bin to *(bin+n)
	// then replace *bout to *(bout+n) by A*vec
	template <typename T>
	void Amul(T* pin, T* pout, int n) const {
		ITensor itx(act_is);
		ITensor ity(act_is);

		Cplx val(1.0, 1.0);
		itx.set(*iterInds(act_is), *(T*)&val);
		ity.set(*iterInds(act_is), *(T*)&val);
		//todo: avoid this memory copying by modifying the allocator of vector_no_init
		// copy the memory to itensor itx, do contraction, then copy memory of ity back
		vector_no_init<T>& dvecx = (*((ITWrap<Dense<T>>*) & (*itx.store()))).d.store;
		dvecx.assign(pin, pin + n);
		this->product(itx, ity);
		vector_no_init<T>& dvecy = (*((ITWrap<Dense<T>>*) & (*ity.store()))).d.store;
		std::memcpy(pout, &dvecy[0], n * sizeof(T));
	}



	//define max-vec product A x -> y
	virtual
	void product(ITensor const& x, ITensor& y) const = 0;
	/*{
		y = A_ * x;
		y.noPrime();
	}*/

};


// for arpack wrapper
namespace itwrap {

#if 0
// If real, then eT == eeT=double; if complex, eT == std::complex<eeT>.
// For real calls, rwork is ignored; it's only necessary in the complex case.
template<typename eT, typename eeT>
inline
void
naupd(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, eeT* tol, eT* resid, blas_int* ncv, eT* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, eT* workd, eT* workl, blas_int* lworkl, eeT* rwork, blas_int* info)
{
	arma_type_check((is_supported_blas_type<eT>::value == false));

	if (is_float<eT>::value) { typedef float     T; arma_ignore(rwork); arma_fortran(arma_snaupd)(ido, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
	else if (is_double<eT>::value) { typedef double    T; arma_ignore(rwork); arma_fortran(arma_dnaupd)(ido, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
	else if (is_cx_float<eT>::value) { typedef cx_float  T; typedef float  xT;  arma_fortran(arma_cnaupd)(ido, bmat, n, which, nev, (xT*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, (xT*)rwork, info); }
	else if (is_cx_double<eT>::value) { typedef cx_double T; typedef double xT;  arma_fortran(arma_znaupd)(ido, bmat, n, which, nev, (xT*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, (xT*)rwork, info); }
}

template<typename eT, typename eeT>
inline
void
saupd(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, eeT* tol, eT* resid, blas_int* ncv, eT* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, eT* workd, eT* workl, blas_int* lworkl, blas_int* info)
{
	arma_type_check((is_supported_blas_type<eT>::value == false));

	if (is_float<eT>::value) { typedef float  T; arma_fortran(arma_ssaupd)(ido, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
	else if (is_double<eT>::value) { typedef double T; arma_fortran(arma_dsaupd)(ido, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
}

template<typename eT>
inline
void
seupd(blas_int* rvec, char* howmny, blas_int* select, eT* d, eT* z, blas_int* ldz, eT* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, eT* tol, eT* resid, blas_int* ncv, eT* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, eT* workd, eT* workl, blas_int* lworkl, blas_int* info)
{
	arma_type_check((is_supported_blas_type<eT>::value == false));

	if (is_float<eT>::value) { typedef float  T; arma_fortran(arma_sseupd)(rvec, howmny, select, (T*)d, (T*)z, ldz, (T*)sigma, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
	else if (is_double<eT>::value) { typedef double T; arma_fortran(arma_dseupd)(rvec, howmny, select, (T*)d, (T*)z, ldz, (T*)sigma, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
}

// for complex versions, pass d for dr, and null for di; pass sigma for
  // sigmar, and null for sigmai; rwork isn't used for non-complex versions
template<typename eT, typename eeT>
inline
void
neupd(blas_int* rvec, char* howmny, blas_int* select, eT* dr, eT* di, eT* z, blas_int* ldz, eT* sigmar, eT* sigmai, eT* workev, char* bmat, blas_int* n, char* which, blas_int* nev, eeT* tol, eT* resid, blas_int* ncv, eT* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, eT* workd, eT* workl, blas_int* lworkl, eeT* rwork, blas_int* info)
{
	arma_type_check((is_supported_blas_type<eT>::value == false));

	if (is_float<eT>::value) { typedef float     T; arma_ignore(rwork); arma_fortran(arma_sneupd)(rvec, howmny, select, (T*)dr, (T*)di, (T*)z, ldz, (T*)sigmar, (T*)sigmai, (T*)workev, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
	else if (is_double<eT>::value) { typedef double    T; arma_ignore(rwork); arma_fortran(arma_dneupd)(rvec, howmny, select, (T*)dr, (T*)di, (T*)z, ldz, (T*)sigmar, (T*)sigmai, (T*)workev, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); }
	else if (is_cx_float<eT>::value) { typedef cx_float  T; typedef float  xT;  arma_fortran(arma_cneupd)(rvec, howmny, select, (T*)dr, (T*)z, ldz, (T*)sigmar, (T*)workev, bmat, n, which, nev, (xT*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, (xT*)rwork, info); }
	else if (is_cx_double<eT>::value) { typedef cx_double T; typedef double xT;  arma_fortran(arma_zneupd)(rvec, howmny, select, (T*)dr, (T*)z, ldz, (T*)sigmar, (T*)workev, bmat, n, which, nev, (xT*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, (xT*)rwork, info); }
}

#endif


inline
void
naupd(int* ido, char* bmat, int* n, char* which, int* nev, double* tol, double* resid, int* ncv, double* v, int* ldv, int* iparam, int* ipntr, double* workd, double* workl, int* lworkl, double* rwork, int* info)
{
	arma_type_check((is_supported_blas_type<double>::value == false));

	typedef double    T;
	arma_ignore(rwork);
	arma_fortran(arma_dnaupd)(ido, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info);
}


// T can be double or complex
template<typename T>
inline
void
naupdT(int* ido, char* bmat, int* n, char* which, int* nev, double* tol, T* resid, int* ncv, T* v, int* ldv, int* iparam, int* ipntr, T* workd, T* workl, int* lworkl, double* rwork, int* info)
{
	arma_type_check((is_supported_blas_type<T>::value == false));

	if (is_double<T>::value) {
		typedef double    Td;
		arma_ignore(rwork);
		arma_fortran(arma_dnaupd)(ido, bmat, n, which, nev, tol, (Td*)resid, ncv, (Td*)v, ldv, iparam, ipntr, (Td*)workd, (Td*)workl, lworkl, info);
	}
	else if (is_cx_double<T>::value) {
		typedef Cplx Tc;
		//typedef double xT; 
		arma_fortran(arma_znaupd)(ido, bmat, n, which, nev, tol, (Tc*)resid, ncv, (Tc*)v, ldv, iparam, ipntr, (Tc*)workd, (Tc*)workl, lworkl, rwork, info);
	}
}


inline
void
dsaupd(int* ido, char* bmat, int* n, char* which, int* nev, double* tol, double* resid, int* ncv, double* v, int* ldv, int* iparam, int* ipntr, double* workd, double* workl, int* lworkl, int* info) {
	arma_fortran(arma_dsaupd)(ido, bmat, n, which, nev, (double*)tol, (double*)resid, ncv, (double*)v, ldv, iparam, ipntr, (double*)workd, (double*)workl, lworkl, info);
}



inline
void
neupd(blas_int* rvec, char* howmny, blas_int* select, double* dr, double* di, double* z, blas_int* ldz, double* sigmar, double* sigmai, double* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, double* rwork, blas_int* info)
{
	arma_type_check((is_supported_blas_type<double>::value == false));

	typedef double    T;
	arma_ignore(rwork);
	arma_fortran(arma_dneupd)(rvec, howmny, select, (T*)dr, (T*)di, (T*)z, ldz, (T*)sigmar, (T*)sigmai, (T*)workev, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info);
}


// for complex versions, pass d for dr, and null for di;
// pass sigma for sigmar, and null for sigmai
// rwork isn't used for non-complex versions
template<typename T>
inline
void
neupdT(blas_int* rvec, char* howmny, blas_int* select, T* dr, T* di, T* z, blas_int* ldz, T* sigmar, T* sigmai, T* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, T* resid, blas_int* ncv, T* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, T* workd, T* workl, blas_int* lworkl, double* rwork, blas_int* info)
{
	arma_type_check((is_supported_blas_type<double>::value == false));

	if (is_double<T>::value) {
		typedef double    Td;
		arma_ignore(rwork);
		arma_fortran(arma_dneupd)(rvec, howmny, select, (Td*)dr, (Td*)di, (Td*)z, ldz, (Td*)sigmar, (Td*)sigmai, (Td*)workev, bmat, n, which, nev, (Td*)tol, (Td*)resid, ncv, (Td*)v, ldv, iparam, ipntr, (Td*)workd, (Td*)workl, lworkl, info);
	}

	else if (is_cx_double<T>::value) {
		typedef cx_double Tc;
		//typedef double xT;
		arma_fortran(arma_zneupd)(rvec, howmny, select, (Tc*)dr, (Tc*)z, ldz, (Tc*)sigmar, (Tc*)workev, bmat, n, which, nev, tol, (Tc*)resid, ncv, (Tc*)v, ldv, iparam, ipntr, (Tc*)workd, (Tc*)workl, lworkl, rwork, info);
	}
}


inline
void
dseupd(blas_int* rvec, char* howmny, blas_int* select, double* d, double* z, blas_int* ldz, double* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info) {
	arma_fortran(arma_dseupd)(rvec, howmny, select, (double*)d, (double*)z, ldz, (double*)sigma, bmat, n, which, nev, (double*)tol, (double*)resid, ncv, (double*)v, ldv, iparam, ipntr, (double*)workd, (double*)workl, lworkl, info);
}



//    eT=T=double
// or eT=double, T=complex
template<typename T>
inline
void
run_aupd
(
	int nev, char* which, const ITensorMapBase& AMap, const bool sym,
	int& n, double& tol,
	T* resid, int& ncv, T* v, int& ldv,
	int* iparam, int* ipntr,
	T* workd, T* workl, int& lworkl, double* rwork,
	int& info
)
{
	// ARPACK provides a "reverse communication interface" which is an
	// entertainingly archaic FORTRAN software engineering technique that
	// basically means that we call saupd()/naupd() and it tells us with some
	// return code what we need to do next (usually a matrix-vector product) and
	// then call it again.  So this results in some type of iterative process
	// where we call saupd()/naupd() many times.
	int ido = 0; // This must be 0 for the first call.
	char bmat = 'I'; // We are considering the standard eigenvalue problem.



	// All the parameters have been set or created.  Time to loop a lot.
	while (ido != 99)
	{
		// Call saupd() or naupd() with the current parameters.
		if (sym) {
			dsaupd(&ido, &bmat, &n, which, &nev, &tol, (double*)resid, &ncv, (double*)v, &ldv, iparam, ipntr, (double*)workd, (double*)workl, &lworkl, &info);
		}
		else {
			naupdT<T>(&ido, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, rwork, &info);
		}

		// What do we do now?
		switch (ido)
		{
		case -1:
			// fallthrough
		case 1:
		{
			// We need to calculate the matrix-vector multiplication y = OP * x
			// where x is of length n and starts at workd(ipntr(0)), and y is of
			// length n and starts at workd(ipntr(1)).

			// operator*(sp_mat, vec) doesn't properly put the result into the
			// right place so we'll just reimplement it here for now...


			//// Set the output to point at the right memory.  We have to subtract
			//// one from FORTRAN pointers...
			//Col<T> out(workd.memptr() + ipntr(1) - 1, n, false /* don't copy */);
			//// Set the input to point at the right memory.
			//Col<T> in(workd.memptr() + ipntr(0) - 1, n, false /* don't copy */);

			//todo: avoid this memory copying by modifying the allocator of vector_no_init
			// copy the memory to itensor it_x, do contraction, then copy memory of y back
#if 0
			dvecx.assign(workd + *ipntr - 1, workd + *ipntr - 1 + n);
			AMap.product(it_x, it_y);
			vector_no_init<T>& dvecy = (*((ITWrap<Dense<T>>*) & (*it_y.store()))).d.store;
			std::memcpy(workd + *(ipntr + 1) - 1, &dvecy[0], n * sizeof(T));
#else
			AMap.Amul<T>(workd + *ipntr - 1, workd + *(ipntr + 1) - 1, n);
#endif

			break;
		}
		case 99:
			// Nothing to do here, things have converged.
			break;
		default:
		{
			return; // Parent frame can look at the value of info.
		}
		}
	}
	// The process has ended; check the return code.
	if ((info != 0) && (info != 1))
	{
		// Print warnings if there was a failure.

		if (sym)
		{
			arma_debug_warn("eigs_sym(): ARPACK error ", info, " in saupd()");
		}
		else
		{
			arma_debug_warn("eigs_gen(): ARPACK error ", info, " in naupd()");
		}

		return; // Parent frame can look at the value of info.
	}

}



template<typename T>
bool
eig_arpack(std::vector<Cplx>& eigval, std::vector<ITensor>& eigvecs, const ITensorMapBase& AMap, Args const& args)
{
	int nev = args.getInt("nev", 1);//number of wanted eigenpairs
	//todo: also pass ncv as a parameter
	int maxiter_ = args.getInt("MaxIter", 300);
	int ncv = args.getInt("ncv", std::max(20, 2 * nev + 1));
	//int maxrestart_ = args.getInt("MaxRestart", 0);
	double tol = args.getReal("tol", 1E-10);
	bool sym = args.getBool("sym", false);
	bool re_eigvec = args.getBool("ReEigvec", false);
	std::string whicheig = args.getString("WhichEig", "LM");

	char* which = const_cast<char*>(whicheig.c_str());


	if (sym) {
		if (is_cx_double<T>::value) {
			Error("Cannot set sym = true for complex operator.");
		}
	}


	int n = AMap.size();

	if (nev + 1 >= n) {
		Error("n_eigval + 1 > matrix size\n");
	}

	//int ncv = nev + 2 + 1;
	if (ncv < nev + 2 + 1) { ncv = nev + 2 + 1; }
	if (ncv < (2 * nev + 1)) { ncv = 2 * nev + 1; }
	if (ncv > n) { ncv = n; }

	int ldv = n;
	int lworkl = 3 * (ncv * ncv) + 6 * ncv;
	int info = 0; //Set to 0 initially to use random initial vector.

	//T* resid, v, workd, workl;

	T* v = new T[n * ncv];
	T* resid = new T[n];
	T* workd = new T[3 * n];
	T* workl = new T[lworkl];

	int iparam[11] = { 0 };
	iparam[0] = 1; // Exact shifts (not provided by us).
	iparam[2] = maxiter_; // Maximum iterations; all the examples use 300.
	iparam[6] = 1; // Mode 1: A * x = lambda * x.

	int ipntr[14] = { 0 };

	//rwork is only used in complex case e.g. cnaupd and znaupd
	double* rwork = new double[ncv]; // Not used in the real case.

	int nev_ = nev;

	run_aupd<T>(nev_, which, AMap, sym, n, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);

	if (info != 0) {
		if (info == 1) std::cout << "Maximum number of iterations taken in run_aupd." << std::endl;
		if (info == -5) std::cout << "WhichEig must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'" << std::endl;
	}


	int rvec = 1; // return vec = true
	char howmny = 'A';
	char bmat = 'I'; // We are considering the standard eigenvalue problem.
	const int ncv_ = ncv;

	int* select = new int[ncv];
	std::memset(select, 0, ncv * sizeof(int));

	T* dr = new T[nev + 1];
	T* di = new T[nev + 1];
	std::memset(dr, 0, (nev + 1) * sizeof(T));
	std::memset(di, 0, (nev + 1) * sizeof(T));

	T* z = new T[n * (nev + 1)];//store the final eigenvectors
	std::memset(z, 0, n * (nev + 1) * sizeof(T));

	int ldz = n;
	T* workev = new T[3 * ncv];

	if (sym) {
		dseupd(&rvec, &howmny, select, (double*)dr, (double*)z, &ldz, (double*)NULL, &bmat, &n, which, &nev, &tol, (double*)resid, &ncv, (double*)v, &ldv, iparam, ipntr, (double*)workd, (double*)workl, &lworkl, &info);
	}
	else {
		neupdT<T>(&rvec, &howmny, select, dr, di, z, &ldz, (T*)NULL, (T*)NULL, workev, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, rwork, &info);
	}

	//todo: give warning about other cases
	if (info != 0)
	{
		std::cout << "\n eigs_gen(): ARPACK error " << info << " in neupd()" << std::endl;
		return false;
	}


	eigval = {};
	eigvecs = {};

	eigval.reserve(nev);
	eigvecs.reserve(nev);

	if (is_double<T>::value) {
		double* dr_ = (double*)dr;
		double* di_ = (double*)di;
		for (int i = 0; i < nev; ++i) {
			eigval.emplace_back(dr_[i], di_[i]);
		}
	}
	else if (is_cx_double<T>::value) {
		for (int i = 0; i < nev; ++i) {
			//Cplx* dr_ = (Cplx*)dr;
			eigval.push_back(dr[i]);
		}
	}
	

	IndexSet act_is = AMap.active_inds();

	if (re_eigvec && is_double<T>::value)
	{
		double* z_ = (double*)(z);
		// reorganize the eigenvectors
		for (int i = 0; i < nev; ++i) {

			vector_no_init<Cplx> veci(n, 0); // store the extracted elements of veci
			vector_no_init<Cplx> veci1(n, 0); // vec i+1

			ITensor Ai(act_is);
			ITensor Ai1(act_is);
			Ai.set(*iterInds(act_is), Cplx(1.0, 1.0));
			Ai1.set(*iterInds(act_is), Cplx(1.0, 1.0));
			vector_no_init<Cplx>& dAi = (*((ITWrap<Dense<Cplx>>*) & (*Ai.store()))).d.store;
			vector_no_init<Cplx>& dAi1 = (*((ITWrap<Dense<Cplx>>*) & (*Ai1.store()))).d.store;

			// i and i+1 is a pair
			if ((i < nev - 1) && (eigval[i] == std::conj(eigval[i + 1]))) {
				for (int j = 0; j < n; ++j)
				{
					veci[j] = Cplx(z_[n * i + j], z_[n * (i + 1) + j]);
					veci1[j] = Cplx(z_[n * i + j], -z_[n * (i + 1) + j]);
				}

				dAi.assign(veci.begin(), veci.end());
				dAi1.assign(veci1.begin(), veci1.end());

				eigvecs.push_back(Ai);
				eigvecs.push_back(Ai1);

				++i; // Skip the next one.
			}

			// if conjugate eigval don't match
			else if ((i == nev - 1) && (Cplx(eigval[i]).imag() != 0.0)) {
				for (int j = 0; j < n; ++j)
				{
					veci[j] = Cplx(z_[n * i + j], z_[n * (i + 1) + j]);
				}

				dAi.assign(veci.begin(), veci.end());

				eigvecs.push_back(Ai);
			}

			// real eigenvalue
			else {
				for (int j = 0; j < n; ++j)
				{
					veci[j] = Cplx(z_[n * i + j], 0);
				}

				dAi.assign(veci.begin(), veci.end());

				eigvecs.push_back(Ai);
			}
		}

	}

	// in complex case, eigenvectors have a direct layout
	if (re_eigvec && is_cx_double<T>::value) {
		for (int i = 0; i < nev; ++i) {
			ITensor Ai(act_is);
			Ai.set(*iterInds(act_is), Cplx(1.0, 1.0));
			vector_no_init<Cplx>& dAi = (*((ITWrap<Dense<Cplx>>*) & (*Ai.store()))).d.store;

			dAi.assign(z + n * i, z + n * (i + 1));
			eigvecs.push_back(Ai);
		}
	}


	//result of dseupd is not sorted
	if (sym && whicheig == "LM") ssort(eigval, eigvecs);


	//clear memory
	delete[] v;
	delete[] resid;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] rwork;
	delete[] dr;
	delete[] di;
	delete[] z;
	delete[] workev;

	return false;
}//eig_arpack



}//namespace it_wrap