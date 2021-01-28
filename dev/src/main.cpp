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



//todo: extract_it for complex

//#include "sample/ctmrg.h"
#include "itensor/all.h"

int main() {
	//#define _ITERATOR_DEBUG_LEVEL 0;

	//constexpr int n = 1000;
	arma::cx_mat M(arma::randn(2,3), arma::randn(2, 3));
	M.print("M");
	auto start = M.begin();
	auto end = M.end();

	std::vector<Cplx> vec(6);
	vec.assign(start, end);

	ITensor A = extract_it(M);
	PrintData(A);
	A.set(1, 1, 10.0);
	M.print("M");
	//PrintData(A);
}

