#pragma once


#include "../pch.h"
#include "../util/database.h"
#include "../util/extract_mat.h"
#include "../util/cft_data.h"
#include "../util/glue.h"


void fibo_exact() {
	int n = 6;

	/*
	ITensor A = database::fibo_vav();
	A.replaceTags("ul", "u");
	A.replaceTags("ur", "r");
	A.replaceTags("lr", "d");
	A.replaceTags("ll", "l");
	*/

	ITensor A = database::fibo_svd_bd();

	ITensor TM = glue(A, n);
	arma::mat TMmat = extract_mat(TM);
	arma::sp_mat TMsparse(TMmat);

	cd_sparse(TMsparse, 5, n);
}