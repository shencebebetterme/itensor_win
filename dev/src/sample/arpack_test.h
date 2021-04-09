#pragma once

#include "pch.h"

#include "inc.h"
#include "util/database.h"
#include "util/extract_mat.h"
#include "util/cft_data.h"
#include "util/glue.h"
#include "util/it_help.h"


//ITensor trg(ITensor, int, int);


void testCFT(ITensor A) {
	printf("\n testing CFT \n");
	Print(A);
	ITensor As = glue(A, 2);
	Print(As);
	arma::mat Asmat = extract_mat(As);
	//Asmat.print("A");
	printf("\n");
	cd_dense(Asmat, 5, 2, 1);
}









//#include "sample/ctmrg.h"
//#include "itensor/all.h"
//#include "util/arnoldi.h"
#include "util/arpack_wrap.h"


class MyITensorMap : public ITensorMapBase {
public:
	ITensor A_;

	// pass the default constructor
	MyITensorMap(IndexSet& is)
		: ITensorMapBase(is)
	{}

	// implement the product interface
	void product(ITensor const& x, ITensor& y) const
	{
		y = A_ * x;
		y.noPrime();
	}
};




void arpack_test() {
	auto i = Index(20, "i");

	//using Type = Cplx;
	using Type = double;

	ITensor A(i, prime(i));
	//A.set(1, 1, Cplx(0, 1.0));
	for (auto a : range1(i.dim()))
		for (auto b : range1(i.dim()))
		{
			double val = std::sin(0.5 * a + (0.3 * a) / b) + 0.01*std::cos(a/b);
			A.set(a, b, val);
		}

	bool isSym = false;

	std::cout << "norm of A is " << norm(A) << std::endl;
	//A += swapTags(A, "0", "1"); isSym = true;
	A /= norm(A);

	int nev = 5;

	//derived from the base class
	//overload the product function
	IndexSet Ais = findInds(A, "0");


	auto AM = MyITensorMap(Ais);
	AM.A_ = A;


	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<Type>(eigval, eigvecs, AM, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true,"sym=",isSym,"WhichEig=","SM" });
	//auto [U, D] = eigen(A);
	//PrintData(D);
	for (int i = 0; i < nev; i++) {
		std::cout << norm(noPrime(A * eigvecs[i]) - eigval[i] * eigvecs[i]) << std::endl;
	}

	for (int i = 0; i < nev; i++) {
		std::cout << "\n eigenpair " << eigval[i] << std::endl;
		PrintData(eigvecs[i]);
	}

	int a = 1;

}



//periodic MPO
class MyITensorMap2 : public ITensorMapBase {
public:
	ITensor A_;
	int N_ = 1;

	// pass the default constructor
	MyITensorMap2(IndexSet& is)
		: ITensorMapBase(is)
	{}

	// implement the product interface A x --> y
	void product(ITensor const& x, ITensor& y) const
	{
		//y = A_ * x;
		//replace all "u" tags to "d" tags
		y = A_ * x;
		for (int i = 1; i < N_; i++) {
			ITensor Ai = prime(A_, i);
			y *= delta(findIndex(y, "r"), findIndex(Ai, "l")) * Ai;
		}
		y *= delta(findIndex(y, "r"), findIndex(y, "l"));
		y.replaceTags("u", "d");
	}
};


void arpack_test2() {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);
	
	Index d = findIndex(A0, "d");

	int N = 20;
	std::vector<Index> idv = {};
	for (int i = 0; i < N; i++) {
		Index di = prime(d, i);
		idv.push_back(di);
	}

	IndexSet is(idv);

	MyITensorMap2 Amap(is);
	const double c_ising = 0.5;//central charge
	const double f_ising = 0.929695398340790;//free energy per site
	const double fA = f_ising * 2; //free energy per A tensor
	Amap.A_ = A0 / std::exp((2 * PI /std::pow(N,2)) * (c_ising / 12) + fA);
	Amap.N_ = N;

	int nev = 10;
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, Amap, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true});

	//show_cd(arma::cx_vec eigval, int num, int n, int m = 1);
	show_cd(eigval, nev, N, 1);
}