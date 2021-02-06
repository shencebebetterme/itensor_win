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

int main() {
	auto i = Index(10, "i");
	//auto j = Index(20, "j");
	//auto k = Index(2, "k");
	//auto A = randomITensor(i, j, k, prime(i), prime(j), prime(k));
	//auto A = randomITensor(i, prime(i));
	
	// 
	//auto At = swapTags(A, "0", "1");
	//A += At;
	//A /= norm(A);
	//auto A = randomITensor(i, prime(i));
	ITensor A(i, prime(i));
	for (auto a : range1(i.dim()))
		for (auto b : range1(i.dim()))
		{
			double val = std::sin(0.5 * a + (0.3 * a) / b);
			A.set(a, b, val);
		}
	//A.randomize();
	//A.set(1, 2, 1,1,1,1, 0.1);
	A /= norm(A);

	auto AM = ITensorMap(A);
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack(eigval, eigvecs, AM, {"NEV=",2,"ErrGoal=1E-5"});
	//auto [U, D] = eigen(A);
	//PrintData(D);

	/*
	auto AM = ITensorMap(A);

	//auto x1 = randomITensor(i, j, k);
	auto x1 = randomITensor(i);
	//x1.set(1, 0.98);
	//x1.set(2, 0.45);
	//auto x2 = randomITensor(i);

	constexpr int nvec = 6;
	std::vector<ITensor> xvec(nvec, x1);

	//xvec = { x1,x2 };

	auto lambda = my_arnoldi(AM, xvec, { "ErrGoal=",1E-10,"MaxIter=",60,"MaxRestart=",5,"Npass=",2 });

	for (auto i : range(nvec)) {
		std::cout << lambda[i] << "\n";
		PrintData(norm((A * xvec[i]).noPrime() - lambda[i] * xvec[i]));
		std::cout << std::endl;
	}
	*/
}

