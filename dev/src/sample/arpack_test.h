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

void arpack_test() {
	auto i = Index(20, "i");

	ITensor A(i, prime(i));
	for (auto a : range1(i.dim()))
		for (auto b : range1(i.dim()))
		{
			double val = std::sin(0.5 * a + (0.3 * a) / b);
			A.set(a, b, val);
		}

	A /= norm(A);

	int nev = 3;

	using T = double;
	auto AM = ITensorMap(A);
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, AM, { "nev=",nev,"ErrGoal=",1E-8, "ReEigvec=",true });
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

