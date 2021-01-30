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



// extract a rank-2 ITensor from an arma::mat
ITensor my_extract_it(arma::mat& M) {
	int nr = M.n_rows;
	int nc = M.n_cols;
	Index i(nr, "i");
	Index j(nc, "j");
	ITensor A = setElt(0.0, i = 1, j = 1);

	vector_no_init<double>& dvec = (*((ITWrap<Dense<double>>*) & (*A.store()))).d.store;
	dvec.assign(M.begin(), M.end());

	return A;
}





//#include "sample/ctmrg.h"
#include "itensor/all.h"

int main() {
	int N = 8;
	auto sites = SpinHalf(N);
	auto indices = sites.inds();

	//auto psi = randomITensor(QN({ "Sz",0 }), indices);

	auto ampo = AutoMPO(sites);
	for (auto j : range1(N - 1))
	{
		ampo += -4, "Sz", j, "Sz", j + 1;
		ampo += -2, "Sx", j;
	}
	ampo += -4, "Sz", N, "Sz", 1;
	ampo += -2, "Sx", N;
	//ampo += "Sz", N, "Sz", 1; //periodic boundary condition?
	auto Hmpo = toMPO(ampo);

	auto H = Hmpo(1);
	for (auto j : range1(2, N)) H *= Hmpo(j);

	H = removeQNs(H);
	Print(H);

	auto [U, D] = eigen(H);
}

