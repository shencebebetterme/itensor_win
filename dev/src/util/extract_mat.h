#pragma once

#include "../inc.h"
using namespace itensor;



// rank-2 itensor to a dense matrix
arma::mat extract_mat(const ITensor&);

arma::cx_mat extract_cxmat(const ITensor& T);

ITensor extract_it(arma::mat& M);
ITensor extract_it(arma::cx_mat& M);


/*
// rank-2 itensor to a sparse matrix
arma::sp_mat extract_spmat(const ITensor&);
*/