#pragma once

#include "../inc.h"
using namespace itensor;



// rank-2 itensor to a dense matrix
arma::mat extract_mat(const ITensor&);


/*
// rank-2 itensor to a sparse matrix
arma::sp_mat extract_spmat(const ITensor&);
*/