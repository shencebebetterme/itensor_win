#pragma once

#include "../inc.h"


constexpr double PI = 3.1415926535;


// cft data from sparse transfer matrix
// show the first num states of the transfer matrix with length n and height m 
void cd_sparse(arma::sp_mat& TM_sparse, int num, int n, int m = 1);

// cft data from dense transfer matrix
// show the first num states of the transfer matrix with length n and height m 
void cd_dense(arma::mat& TM, int num, int n, int m = 1);

// the first argument, cx_vec can also be constructed from std::vector<Cplx>
void show_cd(arma::cx_vec eigval, int num, int n, int m);