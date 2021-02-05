#include "pch.h"
#include "arpack_wrap.h"



// for arpack wrapper
namespace itwrap {

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


#if 0
//    eT=T=double
// or eT=double, T=complex
template<typename T>
inline
void
run_aupd
(
	const uword n_eigvals, char* which, ITensorMap& AMap, const bool sym,
	blas_int& n, double& tol,
	T* resid, blas_int& ncv, T* v, blas_int& ldv,
	blas_int* iparam, blas_int* ipntr,
	T* workd, T* workl, blas_int& lworkl, double* rwork,
	blas_int& info
)
{
	// ARPACK provides a "reverse communication interface" which is an
	// entertainingly archaic FORTRAN software engineering technique that
	// basically means that we call saupd()/naupd() and it tells us with some
	// return code what we need to do next (usually a matrix-vector product) and
	// then call it again.  So this results in some type of iterative process
	// where we call saupd()/naupd() many times.
	blas_int ido = 0; // This must be 0 for the first call.
	char bmat = 'I'; // We are considering the standard eigenvalue problem.
	n = AMap.size(); // The size of the matrix.
	blas_int nev = n_eigvals;

	//resid.set_size(n);
	resid = new T[n];

	// Two contraints on NCV: (NCV > NEV + 2) and (NCV <= N)
	// 
	// We're calling either arpack::saupd() or arpack::naupd(),
	// which have slighly different minimum constraint and recommended value for NCV:
	// http://www.caam.rice.edu/software/ARPACK/UG/node136.html
	// http://www.caam.rice.edu/software/ARPACK/UG/node138.html

	ncv = nev + 2 + 1;

	if (ncv < (2 * nev + 1)) { ncv = 2 * nev + 1; }
	if (ncv > n) { ncv = n; }

	//v.set_size(n * ncv); // Array N by NCV (output).
	v = new T[n * ncv];
	//rwork.set_size(ncv); // Work array of size NCV for complex calls.
	rwork = new T[ncv];

	ldv = n; // "Leading dimension of V exactly as declared in the calling program."

	// IPARAM: integer array of length 11.
	iparam.zeros(11);
	iparam(0) = 1; // Exact shifts (not provided by us).
	iparam(2) = 300; // Maximum iterations; all the examples use 300, but they were written in the ancient times.
	iparam(6) = 1; // Mode 1: A * x = lambda * x.

	// IPNTR: integer array of length 14 (output).
	//ipntr.set_size(14);
	ipntr = new int[14];

	// Real work array used in the basic Arnoldi iteration for reverse communication.
	//workd.set_size(3 * n);
	workd = new T[3 * n];

	// lworkl must be at least 3 * NCV^2 + 6 * NCV.
	lworkl = 3 * (ncv * ncv) + 6 * ncv;

	// Real work array of length lworkl.
	//workl.set_size(lworkl);
	workl = new T[lworkl];

	info = 0; // Set to 0 initially to use random initial vector.

	IndexSet& act_is = AMap.active_inds();
	ITensor it_x(act_is);
	// make the store has type = double or complex
	Cplx val(1.0, 1.0);
	it_x.set(*iterInds(act_is), * (T*)&val);

	vector_no_init<T>& dvecx = (*((ITWrap<Dense<T>>*) & (*it_x.store()))).d.store;

	ITensor it_y = it_x;
	vector_no_init<T>& dvecy = (*((ITWrap<Dense<T>>*) & (*it_y.store()))).d.store;


	//ITensor in = setElt(*IndexValIter(act_is), 1.0);
	// All the parameters have been set or created.  Time to loop a lot.
	while (ido != 99)
	{
		//todo: change the memptr() to normal pointer
		// Call saupd() or naupd() with the current parameters.
		if (sym)
			saupd(&ido, &bmat, &n, which, &nev, &tol, resid.memptr(), &ncv, v.memptr(), &ldv, iparam.memptr(), ipntr.memptr(), workd.memptr(), workl.memptr(), &lworkl, &info);
		else
			naupd(&ido, &bmat, &n, which, &nev, &tol, resid.memptr(), &ncv, v.memptr(), &ldv, iparam.memptr(), ipntr.memptr(), workd.memptr(), workl.memptr(), &lworkl, rwork.memptr(), &info);

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

			//todo: overload this mat-vec product
			
			//// Set the output to point at the right memory.  We have to subtract
			//// one from FORTRAN pointers...
			//Col<T> out(workd.memptr() + ipntr(1) - 1, n, false /* don't copy */);
			//// Set the input to point at the right memory.
			//Col<T> in(workd.memptr() + ipntr(0) - 1, n, false /* don't copy */);
			
			// copy the memory to itensor it_x, do contraction, then copy memory of y back
			dvecx.assign(workd.memptr() + ipntr(0) - 1, workd.memptr() + ipntr(0) - 1 + n);
			AMap.product(it_x, it_y);
			std::memcpy(workd.memptr() + ipntr(1) - 1, dvecy.begin(), n * sizeof(T));


			//out.zeros();
			//typename SpMat<T>::const_iterator x_it = X.begin();
			//typename SpMat<T>::const_iterator x_it_end = X.end();

			////implement the mat-vec product of a sparse matrix
			//while (x_it != x_it_end)
			//{
			//	out[x_it.row()] += (*x_it) * in[x_it.col()];
			//	++x_it;
			//}

			// No need to modify memory further since it was all done in-place.

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

	// clear memory
	delete[] resid;
	delete[] v;
	delete[] rwork;
	delete[] ipntr;
	delete[] workd;
	delete[] workl;
}
#endif


//template<typename T>
inline
bool
eig_arpack(std::vector<Cplx>& eigval, std::vector<ITensor>& eigvecs, const ITensorMap& AMap, Args const& args)
{
	int nev_ = args.getInt("NEV", 1);
	int maxiter_ = args.getInt("MaxIter", 10);
	int maxrestart_ = args.getInt("MaxRestart", 0);
	const double errgoal_ = args.getReal("ErrGoal", 1E-6);
	bool symm_ = args.getBool("RealSymmetric", false);
	char* which = const_cast<char*>(args.getString("WhichEig", "LM").c_str());

	if (nev_ + 1 >= AMap.size()) {
		itensor::error("n_eigval + 1 > matrix size\n");
	}

	//todo: calculate the needed space and allocate memory before calling run_aupd
	int n, ncv, ldv, lworkl, info;
	double tol = errgoal_;
	double* resid, v, workd, workl;
	int iparam[11], ipntr[14];
	double* rwork; // Not used in the real case.

	//todo: run_aupd


	return false;
}

}//namespace it_wrap