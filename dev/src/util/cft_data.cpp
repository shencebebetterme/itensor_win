
#include "cft_data.h"
#include <iterator>


// obtain cft data from eigenvalues of transfer matrix
void show_cd(arma::cx_vec eigval, int num, int n, int m = 1) {
    std::vector<double> log_lambdas = {};
    std::vector<double> scaling_dims = {};
    std::vector<double> spins = {};
    //obtain spins
    for (auto lambda_i_tilde : eigval) {
        //eigen_vals.push_back(lambda_i_tilde);
        Cplx log_lambda_i_tilde = std::log(lambda_i_tilde);
        log_lambdas.push_back(log_lambda_i_tilde.real());
        spins.push_back(n / (2 * PI) * log_lambda_i_tilde.imag());
    }
    //obtain scaling dimensions
    double log_lambda_max = log_lambdas[0];
    for (const auto& log_lambda_i : log_lambdas) {
        double i_th_scaling_dim = -n / (2 * m * PI) * (log_lambda_i - log_lambda_max);
        scaling_dims.push_back(i_th_scaling_dim);
    }
    //print
    for (int i = 0; i < num; i++) {
        printf("%6.6f\t%6.6f\n", scaling_dims[i], spins[i]);
    }
}


void cd_sparse(arma::sp_mat& TM_sparse, int num, int n, int m) {
    int nT = TM_sparse.n_rows;
    if (nT < num) {
        printf("\nToo many states requested!\n");
    }

    arma::eigs_opts opt;
    opt.tol = 0.001;
    cx_vec eigval = eigs_gen(TM_sparse, num, "lm", opt);

    show_cd(eigval, num, n, m);
}


void cd_dense(arma::mat& TM, int num, int n, int m) {
    int nT = TM.n_rows;
    if (nT < num) {
        printf("\nToo many states requested!\n");
    }

    //printf("\ncalculating eigenvalues\n");
    cx_vec eigval = eig_gen(TM);
    //printf("eigenvalues calculated\n\n");

    // obtain the first n eigenvalues with largest modulus   
    std::sort(eigval.begin(), eigval.end(),
        [](Cplx& a, Cplx& b) {return std::abs(a) > std::abs(b); }
        );

    //cx_vec eigval_cut(eigval.begin(), std::advance(eigval.begin(), n));

    cx_vec eigval_cut(eigval.memptr(), num);
 
    show_cd(eigval_cut, num, n, m);
}
