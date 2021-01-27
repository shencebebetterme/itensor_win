#pragma once

#include "../inc.h"
#include "../util/database.h"
#include "../util/extract_mat.h"
#include "../util/cft_data.h"
#include "../util/glue.h"
#include "../util/it_help.h"
		  
#include "../TRG/tnr.h"
#include "../TRG/trg.h"
#include "../TRG/GiltTNR.h"
		  
#include "../opt/descend.h"

#include <chrono>
using namespace std::chrono;

//ITensor trg(ITensor, int, int);



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


void test_copy() {
	constexpr int n = 10000;
	Index i(n, "i");
	Index j(n, "j");
	ITensor A = randomITensor(i, j);
	//PrintData(A);

	auto start = high_resolution_clock::now();
	arma::mat Amat1 = extract_mat(A);
	//Amat1(1, 1) = 1.0;
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "extract_mat " << duration.count() / 1000000.0 << std::endl;

	start = high_resolution_clock::now();
	arma::mat Amat2 = extract_mat(A, true);
	(*Amat2)(1, 1) = 1.0;
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << "extract_mat_nocopy true " << duration.count() / 1000000.0 << std::endl;

	start = high_resolution_clock::now();
	arma::mat Amat3 = extract_mat(A, false);
	(*Amat3)(1, 1) = 1.0;
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << "extract_mat_nocopy false " << duration.count() / 1000000.0 << std::endl;


}

