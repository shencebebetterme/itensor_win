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
	arma::mat* Asmat = extract_mat(As);
	//Asmat.print("A");
	printf("\n");
	cd_dense(*Asmat, 5, 2, 1);
}




arma::cx_mat* my_extract_cxmat(const ITensor& T, bool copy) {
	if (!isComplex(T)) {
		std::cerr << "\nTensor not complex!";
		std::abort();
	}

	auto di = T.index(1).dim();
	auto dj = T.index(2).dim();

	auto pt = &((*((ITWrap<Dense<Cplx>>*) & (*T.store()))).d.store[0]);

	arma::cx_mat* denseT = new arma::cx_mat(pt, di, dj, copy);

	return denseT;
}


int main() {
	//constexpr int n = 10000;
	Index i(2, "i");
	Index j(3, "j");
	ITensor A = randomITensorC(i, j);
	PrintData(A);

	arma::cx_mat& mat2 = *my_extract_cxmat(A, true);
	(mat2)(0, 0) = 2.0+2.0i;
	(mat2).print(); PrintData(A);

	arma::cx_mat& mat3 = *my_extract_cxmat(A, false);
	(mat3)(0, 0) = 3.0;
	(mat3).print(); PrintData(A);
}

