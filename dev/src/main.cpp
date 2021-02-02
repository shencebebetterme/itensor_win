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
	auto i = Index(2, "i");
	auto j = Index(2, "j");
	auto k = Index(2, "k");
	auto A = randomITensor(i, j, k, prime(i), prime(j), prime(k));

	auto [U, D] = eigen(A);
	PrintData(D);

	auto AM = ITensorMap(A);

	auto x1 = randomITensor(i, j, k);

	constexpr int nvec = 4;
	std::vector<ITensor> xvec(nvec, x1);

	auto lambda = my_arnoldi(AM, xvec, { "ErrGoal=",1E-10,"MaxIter=",20,"MaxRestart=",5,"Npass=",2 });

	for (auto i : range(nvec)) {
		std::cout << lambda[i] << "\n";
		PrintData(norm((A * xvec[i]).noPrime() - lambda[i] * xvec[i]));
	}

}

