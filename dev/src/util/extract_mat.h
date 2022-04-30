#pragma once

#include "../inc.h"



// rank-2 itensor to a dense matrix
// if copy=false then change in mat will be reflected in T
// usage: arma::mat& m = *extract_mat(A);
// usage: arma::mat* m = extract_mat(A);
arma::mat extract_mat(ITensor& T, bool copy = true);
arma::cx_mat extract_cxmat(ITensor& T, bool copy = true);

ITensor extract_it(const arma::mat& M);
ITensor extract_it(const arma::cx_mat& M);


/*
// rank-2 itensor to a sparse matrix
arma::sp_mat extract_spmat(const ITensor&);
*/