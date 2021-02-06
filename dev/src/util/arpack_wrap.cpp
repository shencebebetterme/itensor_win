#include "pch.h"
#include "arpack_wrap.h"



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

inline
void
neupd(blas_int* rvec, char* howmny, blas_int* select, double* dr, double* di, double* z, blas_int* ldz, double* sigmar, double* sigmai, double* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, double* rwork, blas_int* info)
{
	arma_type_check((is_supported_blas_type<double>::value == false));

	 typedef double    T;
	 arma_ignore(rwork); 
	 arma_fortran(arma_dneupd)(rvec, howmny, select, (T*)dr, (T*)di, (T*)z, ldz, (T*)sigmar, (T*)sigmai, (T*)workev, bmat, n, which, nev, (T*)tol, (T*)resid, ncv, (T*)v, ldv, iparam, ipntr, (T*)workd, (T*)workl, lworkl, info); 
}





//    eT=T=double
// or eT=double, T=complex
template<typename T>
inline
void
run_aupd
(
	int nev, char* which, const ITensorMap& AMap, const bool sym,
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
	


	IndexSet act_is = AMap.active_inds();
	ITensor it_x(act_is);
	// make the store has type = double or complex
	Cplx val(1.0, 1.0);
	it_x.set(*iterInds(act_is), * (T*)&val);

	vector_no_init<T>& dvecx = (*((ITWrap<Dense<T>>*) & (*it_x.store()))).d.store;

	ITensor it_y = it_x;
	vector_no_init<T>& dvecy = (*((ITWrap<Dense<T>>*) & (*it_y.store()))).d.store;


	// All the parameters have been set or created.  Time to loop a lot.
	while (ido != 99)
	{
		// Call saupd() or naupd() with the current parameters.
		//if (sym)
		//	saupd(&ido, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
		//else
			naupd(&ido, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, rwork, &info);

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
			dvecx.assign(workd + *ipntr - 1, workd + *ipntr - 1 + n);
			AMap.product(it_x, it_y);
			std::memcpy(workd + *(ipntr+1) - 1, &dvecy[0], n * sizeof(T));

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


//template<typename T>
bool
eig_arpack(std::vector<Cplx>& eigval, std::vector<ITensor>& eigvecs, const ITensorMap& AMap, Args const& args)
{
	int nev = args.getInt("NEV", 1);//number of wanted eigenpairs
	int maxiter_ = args.getInt("MaxIter", 10);
	int maxrestart_ = args.getInt("MaxRestart", 0);
	double tol = args.getReal("ErrGoal", 1E-4);
	bool sym = args.getBool("RealSymmetric", false);
	std::string whicheig = args.getString("WhichEig", "LM");

	char* which = const_cast<char*>(whicheig.c_str());

	

	//todo: calculate the needed space and allocate memory before calling run_aupd
	
	int n = AMap.size();

	if (nev + 1 >= n) {
		Error("n_eigval + 1 > matrix size\n");
	}

	int ncv = nev + 2 + 1;

	if (ncv < (2 * nev + 1)) { ncv = 2 * nev + 1; }
	if (ncv > n) { ncv = n; }

	int ldv = n;
	int lworkl = 3 * (ncv * ncv) + 6 * ncv;
	int info = 0; //Set to 0 initially to use random initial vector.

	//double* resid, v, workd, workl;

	double* v = new double[n * ncv];
	double* resid = new double[n];
	double* workd = new double[3 * n];
	double* workl = new double[lworkl];
	
	int iparam[11] = { 0 };
	iparam[0] = 1; // Exact shifts (not provided by us).
	iparam[2] = maxiter_; // Maximum iterations; all the examples use 300.
	iparam[6] = 1; // Mode 1: A * x = lambda * x.

	int ipntr[14] = { 0 };

	double* rwork = NULL; // Not used in the real case.


	run_aupd<double>(nev, which, AMap, sym, n, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork,info);

	if (info != 0) return false;//info=0 means normal exit of aupd


	//todo: call neupd to obtain the actual eigenpairs
	int rvec = 1;
	char howmny = 'A';
	char bmat = 'I'; // We are considering the standard eigenvalue problem.
	const int ncv_ = ncv;

	int* select = new int[ncv];
	std::memset(select, 0, ncv * sizeof(int));

	double* dr = new double[nev + 1];
	double* di = new double[nev + 1];
	std::memset(dr, 0, (nev + 1) * sizeof(double));
	std::memset(di, 0, (nev + 1) * sizeof(double));

	double* z = new double[n * (nev + 1)];
	std::memset(z, 0, n * (nev + 1) * sizeof(double));

	int ldz = n;
	double* workev = new double[3 * nev];

	neupd(&rvec, &howmny, select, dr, di, z, &ldz, (double*)NULL, (double*)NULL, workev, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, rwork, &info);

	//todo: reorganize the eigenpairs into eigval and eigvecs

	//clear memory
	delete[] v;
	delete[] resid;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] dr;
	delete[] di;
	delete[] z;
	delete[] workev;

	return false;
}//eig_arpack



}//namespace it_wrap