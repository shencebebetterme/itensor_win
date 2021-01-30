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
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);
	int len = 3;
	ITensor M = glue_bare_ring(A0, len);
	Print(M);
	//M.noPrime();
	for (auto i = 0; i < len; i++) {
		M.replaceTags("u," + str(i), "1,s=" + str(i));
		M.replaceTags("d," + str(i), "0,s=" + str(i));
	}
	//M.replaceTags("u,0", "1");
	//M.replaceTags("d,0", "0");
	Print(M);
	/*IndexSet u_is = findInds(M, "1");
	IndexSet d_is = findInds(M, "0");
	auto [uT, U] = combiner(u_is);
	auto [dT, D] = combiner(d_is);*/

	auto [U, D] = diagHermitian(M);
	PrintData(D);
	PrintData(U);

	
}

