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


//copy once
arma::mat* my_extract_mat1(const ITensor& T) {
	auto di = T.index(1).dim();
	auto dj = T.index(2).dim();

	auto extractReal = [](Dense<Real> const& d)
	{
		return d.store;
	};

	//data already copied to data_vec
	auto data_vec = applyFunc(extractReal, T.store());

	arma::mat* denseT = new arma::mat(&data_vec[0], di, dj, false);
	return denseT;
}


// no copy
arma::mat* my_extract_mat2(ITensor& T) {
	auto di = T.index(1).dim();
	auto dj = T.index(2).dim();

	auto pt = &((*((ITWrap<Dense<double>>*) & (*T.store()))).d.store[0]);
	arma::mat* denseT = new arma::mat(pt, 2, 3, false);
	return denseT;
}


int main() {
	//constexpr int n = 10000;
	Index i(2, "i");
	Index j(3, "j");
	ITensor A = randomITensor(i, j);
	PrintData(A);

	arma::mat* mat2 = extract_mat(A, true);
	(*mat2)(0, 0) = 2.0;
	(*mat2).print(); PrintData(A);

	arma::mat* mat3 = extract_mat(A, false);
	(*mat3)(0, 0) = 3.0;
	(*mat3).print(); PrintData(A);
}

