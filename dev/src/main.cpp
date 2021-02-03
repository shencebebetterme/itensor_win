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
#include "util/arnoldi.h"

int main() {
	auto i = Index(4, "i");
	auto j = Index(5, "j");
	auto k = Index(3, "k");
	auto A = randomITensor(i, j, k, prime(i), prime(j), prime(k));
	//auto A = randomITensor(i, prime(i));
	//for (auto a : range1(i.dim()))
	//	for (auto b : range1(i.dim()))
	//		A.set(a, b, 0);
	//A.set(1, 1, 100.0);
	////A.set(1, 2, 0);
	////A.set(2, 1, 0);
	//A.set(2, 2, 6.0);
	//A.set(3, 3, 6.0);
	//A.set(4, 4, -3.0);

	auto [U, D] = eigen(A);
	PrintData(D);

	auto AM = ITensorMap(A);

	auto x1 = randomITensor(i, j, k);
	//auto x1 = randomITensor(i);
	//x1.set(1, 0.98);
	//x1.set(2, 0.45);
	//auto x2 = randomITensor(i);

	constexpr int nvec = 6;
	std::vector<ITensor> xvec(nvec, x1);

	//xvec = { x1,x2 };

	auto lambda = my_arnoldi(AM, xvec, { "ErrGoal=",1E-14,"MaxIter=",100,"MaxRestart=",10,"Npass=",2 });

	for (auto i : range(nvec)) {
		std::cout << lambda[i] << "\n";
		PrintData(norm((A * xvec[i]).noPrime() - lambda[i] * xvec[i]));
		std::cout << std::endl;
	}

}

