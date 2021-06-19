#include "pch.h"

#include "inc.h"
#include "util/database.h"
#include "util/extract_mat.h"
#include "util/cft_data.h"
#include "util/glue.h"
#include "util/it_help.h"

#include "TRG/tnr.h"
#include "TRG/trg.h"
#include "TRG/GiltTNR.h"

#include "opt/descend.h"

#include <chrono>
using namespace std::chrono;

//ITensor trg(ITensor, int, int);

//#include "sample/ctmrg.h"
//#include "itensor/all.h"
//#include "util/arnoldi.h"
//
#include "util/arpack_wrap.h"
#include "sample/arpack_test.h"
//#include "sample/adjoint.h"

template<typename T>
class MyClass
{
public:
	T x_;
	MyClass(T x) {
		x_ = x;
	}

private:
};

template<typename T, typename T2>
T2 f(MyClass<T>& mc_) {
	mc_.x_ = 0;
	T2 z = mc_.x_;
	return z;
}


void check_ising_spectrum() {
	int N = 12;
	//int N = std::atoi(argv[1]);
	//auto sites = SpinHalf(N);
	auto sites = SpinHalf(N, { "ConserveQNs=",false });

	double h = 1.0; //at critical point

	auto ampo = AutoMPO(sites);

	for (int j = 1; j < N; ++j)
	{
		ampo += -4.0, "Sz", j, "Sz", j + 1;
	}

	ampo += -4.0, "Sz", 1, "Sz", N;

	for (int j = 1; j <= N; ++j) {
		ampo += -2.0 * h, "Sx", j;
	}

	auto H = toMPO(ampo);

	ITensor HT = H.A(1);
	for (int i = 2; i <= N; ++i) {
		HT *= H.A(i);
	}

	HT *= N / (2 * PI);

	//todo: remove the free energy part

	HT *= 0.5; // now HT should have the same spectrum as L0 + L0bar with a shift

	//done: check spectrum of HT
	IndexSet HTis = findInds(HT, "0");
	auto HTM = MyITensorMap(HTis);
	HTM.A_ = HT;

	int nev = 5;
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, HTM, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true,"WhichEig=","SR" });
	//auto [U, D] = eigen(A);
	//PrintData(D);
	for (int i = 0; i < nev; i++) {
		//std::cout << norm(noPrime(HT * eigvecs[i]) - eigval[i] * eigvecs[i]) << std::endl;
	}

	printfln("relative energy spectrum is");
	for (int i = 0; i < nev; i++) {
		std::cout << eigval[i] - eigval[0] << std::endl;
		//PrintData(eigvecs[i]);
	}
}


//MPO chain, with pairs of unprimed and primed indices
class MPOMap : public ITensorMapBase {
public:
	MPO mpo;
	int N_ = 1;// will be set to the length of mpo

	// pass the default constructor
	MPOMap(IndexSet& is)
		: ITensorMapBase(is)
	{}

	void product(ITensor const& x, ITensor& y) const
	{
		//y = A_ * x;
		//replace all "u" tags to "d" tags
		y = mpo.A(1) * x;
		for (int i = 2; i <= N_; i++) {
			y *= mpo.A(i);
		}
		y.noPrime();
	}
};


void check_ising_mpo_spectrum() {
	int N = 16;
	auto sites = SpinHalf(N, { "ConserveQNs=",false });

	double h = 1.0; //at critical point

	auto ampo = AutoMPO(sites);

	for (int j = 1; j < N; ++j)
	{
		ampo += -4.0, "Sz", j, "Sz", j + 1;
	}

	ampo += -4.0, "Sz", 1, "Sz", N;

	for (int j = 1; j <= N; ++j) {
		ampo += -2.0 * h, "Sx", j;
	}

	auto H = toMPO(ampo);

	H *= N / (2 * PI);
	H *= 0.5;// remove an extra factor of 2

	IndexSet HTis = sites.inds();

	MPOMap mmap(HTis);
	mmap.mpo = H;
	mmap.N_ = N;


	int nev = 20;
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, mmap, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true,"WhichEig=","SR" });

	/*for (int i = 0; i < nev; i++) {
		std::cout << eigval[i] << std::endl;
	}*/

	printfln("ground state energy is %.10f", eigval[0]);

	printfln("normalized relative energy spectrum is");
	for (int i = 0; i < nev; i++) {
		double ei = real(eigval[i] - eigval[0]);
		std::cout << ei << std::endl;
		//PrintData(eigvecs[i]);
	}

}


//contract a MPO to an ITensor
//n is the number of shared indices between M and A
ITensor cMT(MPO M, ITensor A, int n) {
	ITensor res = M.A(1) * A;
	for (int i = 2; i <= n; ++i) {
		res *= M.A(i);
	}
	return res;
}


class adMPOMap : public ITensorMapBase {
public:

	MPO Z_; // to implement ad_Z
	int ldim_; // dim of each leg
	int nl_; // Z has 2*nl legs
	IndexSet uis;
	IndexSet dis;
	IndexSet tis; //temporary index sets

	// inds(Z_) = inds(x) = is = {u, u', u'', ... d, d', d'', ...}
	adMPOMap(IndexSet& is)
		: ITensorMapBase(is)
	{}


	// ad_Z maps X to [Z,X]
	void ad(ITensor const& x, ITensor& y) const
	{
		// y = Z x - x Z
		// suppose x and Z has the same indexing structure
		// with indices u1, u2, u3, ... and u1', u2', u3', ...
		// x is an itensor, z is a MPO
		

		MPO Zc = prime(Z_);
		ITensor xc = prime(x);
		y = cMT(Zc, x, nl_) - cMT(Z_, xc, nl_); 
		//Zc.replaceTags("u", "uu").replaceTags("d", "u");
		//xc.replaceTags("u", "uu").replaceTags("d", "u");
		//y = Zc * x - xc * Z_;
		y.replaceTags("2", "1");//now y has indices u1, u2, u3, ... and u1', u2', u3', ...
	}

	// ad^2
	void product(ITensor const& x, ITensor& y) const
	{
		ITensor tmp = x;
		ad(x, tmp);
		ad(tmp, y);
		//y *= -1;
	}
};


double gsE(IndexSet HTis, MPO H, int N) {
	
	MPOMap mmap(HTis);
	mmap.mpo = H;
	mmap.N_ = N;

	int nev = 2;
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, mmap, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true,"WhichEig=","SR" });
	double gse = eigval[0].real();

	return gse;
}



void diag_ad() {
	
	int N = 4;
	auto sites = SpinHalf(N, { "ConserveQNs=",false });

	double h = 1.0; //at critical point

	auto ampo = AutoMPO(sites);

	for (int j = 1; j < N; ++j)
	{
		ampo += -4.0, "Sz", j, "Sz", j + 1;
	}

	ampo += -4.0, "Sz", 1, "Sz", N;

	for (int j = 1; j <= N; ++j) {
		ampo += -2.0 * h, "Sx", j;
	}

	auto H = toMPO(ampo);

	// scale H to L0 + L0bar
	H *= N / (2 * PI);
	H *= 0.5; // remove an extra factor of 2
	//todo: shift H s.t. its lowest eigenvalue is 0
	double gse = gsE(sites.inds(), H, N);
	
	//IndexSet HDis = sites.inds();
	//IndexSet HUis = sites.inds().prime();
	// IndexSet HTis = HDis + HUis
	IndexSet HTis = unionInds(sites.inds(), sites.inds().prime());
	
	adMPOMap admap(HTis);
	admap.Z_ = H;
	admap.nl_ = N;

	int nev = 10;
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, admap, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true,"WhichEig=","SR" });

	printfln("first eigenvalue is %.10f", eigval[0]);

	for (int i = 0; i < nev; i++) {
		double ei = real(eigval[i] - eigval[0]);
		std::cout << ei << std::endl;
		//PrintData(eigvecs[i]);
	}
}


int main() {
	diag_ad();
}