#include "extract_mat.h"

/*
arma::sp_mat extract_spmat(const ITensor& T) {
    auto di = T.index(1).dim();
    auto dj = T.index(2).dim();

    auto extractReal = [](Dense<Real> const& d)
    {
        return d.store;
    };

    auto data_vec = applyFunc(extractReal, T.store());

    arma::mat denseT(&data_vec[0], di, dj, false);
    arma::sp_mat sparseT(denseT);
    return sparseT;
}
*/


arma::mat extract_mat(const ITensor& T) {
    auto di = T.index(1).dim();
    auto dj = T.index(2).dim();

    auto extractReal = [](Dense<Real> const& d)
    {
        return d.store;
    };

    auto data_vec = applyFunc(extractReal, T.store());

    arma::mat denseT(&data_vec[0], di, dj, true);
    return denseT;
}